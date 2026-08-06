import os
import time
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import copy
from typing import Dict, Any, Optional

from runtime_configuration_engine import parse_args
from dataset import MultiKineticsDataset
from core_module import AAPM, compute_dtw_distance, compute_soft_dtw
from enterprise_system_utility_telemetry_diagnostic_suite import (
    EnterpriseTelemetryDiagnosticRegistry, 
    LoggerProxy, 
    MetricTracker, 
    Timer, 
    count_model_parameters, 
    save_json_metadata
)

class ModelEmaV2(nn.Module):
    """Exponential Moving Average wrapper maintaining synchronized shadow weights for stability."""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, model_param in zip(self.module.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_param = model_param.to(device=self.device)
            ema_param.copy_(self.decay * ema_param + (1.0 - self.decay) * model_param)

class LinearWarmupCosineScheduler:
    """Learning rate scheduler implementing linear warmup followed by cosine annealing decay."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            current_lr = self.base_lr * (float(self.current_step) / float(self.warmup_steps))
        else:
            progress_ratio = float(self.current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            current_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + torch.cos(torch.tensor(progress_ratio * 3.14159265)))
            current_lr = current_lr.item()
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

class AAPMTrainerOrchestratorEngine:
    """Enterprise-grade distributed training orchestrator engine managing complete lifecycle execution."""
    def __init__(self, args: Any):
        self.args = args
        self.local_rank = self._setup_distributed_process_group()
        self.is_master = (self.local_rank == 0)
        
        self.logger = LoggerProxy(self.local_rank, args.log_dir)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        self._log_diagnostic_preflight_info()
        self._build_environment_configurations()
        self._build_neural_network_model()
        self._build_optimizer_and_scheduler()
        self._build_distributed_dataloaders()

    def _setup_distributed_process_group(self) -> int:
        if "LOCAL_RANK" in os.environ:
            os.environ["NCCL_DEBUG"] = self.args.nccl_debug
            communication_timeout = datetime.timedelta(seconds=self.args.nccl_timeout)
            dist.init_process_group(backend=self.args.ddp_backend, timeout=communication_timeout)
            local_rank_id = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank_id)
            return local_rank_id
        return 0

    def _log_diagnostic_preflight_info(self) -> None:
        if self.is_master:
            system_metadata = EnterpriseTelemetryDiagnosticRegistry.capture_system_environment_metadata()
            self.logger.info(f"System Pre-flight Diagnostics Captured Successfully: {system_metadata}")

    def _build_environment_configurations(self) -> None:
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.args.seed + self.local_rank)
        
    def _build_neural_network_model(self) -> None:
        with Timer("NeuralNetworkModelInitialization", self.logger):
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.backbone)
            base_model = AAPM(self.args).to(self.device)
            
            parameter_statistics = count_model_parameters(base_model)
            self.logger.info(f"Model Parameter Breakdown Statistics -> Total: {parameter_statistics['total']:,} | Trainable: {parameter_statistics['trainable']:,} | Frozen: {parameter_statistics['frozen']:,}")
            
            if dist.is_initialized():
                self.model = DDP(base_model, device_ids=[self.local_rank], find_unused_parameters=True)
            else:
                self.model = base_model
                
            self.model_ema = ModelEmaV2(self.model, decay=0.9999, device=self.device) if self.args.use_ema else None
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).to(self.device)
            self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=(self.args.mixed_precision == "fp16"))

    def _build_optimizer_and_scheduler(self) -> None:
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.999)
        )

    def _build_distributed_dataloaders(self) -> None:
        self.train_dataset = MultiKineticsDataset(self.args, is_train=True)
        self.val_dataset = MultiKineticsDataset(self.args, is_train=False)
        
        self.train_sampler = DistributedSampler(self.train_dataset) if dist.is_initialized() else None
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if dist.is_initialized() else None
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size_support + self.args.batch_size_query,
            sampler=self.train_sampler, num_workers=self.args.num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size_support + self.args.batch_size_query,
            sampler=self.val_sampler, num_workers=self.args.num_workers, pin_memory=True, drop_last=False
        )
        
        total_training_steps = len(self.train_loader) * self.args.epochs
        warmup_step_count = int(total_training_steps * self.args.warmup_ratio)
        self.scheduler = LinearWarmupCosineScheduler(self.optimizer, warmup_step_count, total_training_steps, self.args.lr)

    def _process_batch_forward_pass(self, video_frames: torch.Tensor, text_inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        constrained_features = self.model(video_frames, text_inputs)
        support_features = constrained_features[:self.args.batch_size_support]
        query_features = constrained_features[self.args.batch_size_support:]
        
        if query_features.shape[0] == 0:
            return None
            
        prototypes = support_features.mean(dim=0, keepdim=True)
        distances = compute_soft_dtw(prototypes, query_features) if self.args.distance_metric == "soft_dtw" else compute_dtw_distance(prototypes, query_features)
            
        logits = -distances / self.args.temperature
        target_labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return self.criterion(logits, target_labels)

    def execute_training_epoch(self, epoch_index: int) -> None:
        self.model.train()
        loss_tracker_meter = MetricTracker()
        
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch_index)
            
        self.optimizer.zero_grad()
        
        for batch_index, (video_batch, label_batch) in enumerate(self.train_loader):
            video_batch = video_batch.to(self.device, non_blocking=True)
            text_prompt_tokens = self.tokenizer(["action_a", "action_b", "action_c", "action_d", "action_e"], padding=True, return_tensors="pt", truncation=True).to(self.device)
            
            mixed_precision_dtype = torch.bfloat16 if self.args.mixed_precision == "bf16" else torch.float16
            with torch.cuda.amp.autocast(enabled=(self.args.mixed_precision != "fp32"), dtype=mixed_precision_dtype):
                batch_loss = self._process_batch_forward_pass(video_batch, text_prompt_tokens)
                if batch_loss is None:
                    continue
                batch_loss = batch_loss / self.args.grad_accum_steps

            self.gradient_scaler.scale(batch_loss).backward()
            
            if (batch_index + 1) % self.args.grad_accum_steps == 0:
                if self.args.clip_grad_norm > 0.0:
                    self.gradient_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                if self.model_ema is not None:
                    self.model_ema.update(self.model)

            loss_tracker_meter.update(batch_loss.item() * self.args.grad_accum_steps, video_batch.size(0))
            
            if self.is_master and batch_index % self.args.log_interval == 0:
                self.logger.info(f"Epoch Training Iteration [{epoch_index}][{batch_index}/{len(self.train_loader)}] Step Loss: {loss_tracker_meter.val:.4f} (Running Average: {loss_tracker_meter.avg:.4f})")

    @torch.no_grad()
    def execute_validation_pass(self) -> float:
        evaluation_model = self.model_ema.module if self.model_ema else self.model
        evaluation_model.eval()
        validation_loss_tracker = MetricTracker()
        
        for video_batch, label_batch in self.val_loader:
            video_batch = video_batch.to(self.device, non_blocking=True)
            text_prompt_tokens = self.tokenizer(["action_a", "action_b", "action_c"], padding=True, return_tensors="pt").to(self.device)
            
            with torch.cuda.amp.autocast(enabled=(self.args.mixed_precision != "fp32")):
                batch_loss = self._process_batch_forward_pass(video_batch, text_prompt_tokens)
                if batch_loss is not None:
                    validation_loss_tracker.update(batch_loss.item(), video_batch.size(0))
                    
        validation_loss_tracker.sync(self.device)
        if self.is_master:
            self.logger.info(f"==> Global Validation Loss Evaluation Score: {validation_loss_tracker.avg:.4f}")
        return validation_loss_tracker.avg

    def fit(self) -> None:
        best_validation_loss = float('inf')
        self.logger.info(f"Launching distributed training routine across total {self.args.epochs} configured epochs.")
        
        if self.is_master:
            save_json_metadata(os.path.join(self.args.checkpoint_dir, "enterprise_experiment_metadata.json"), vars(self.args))

        try:
            for epoch_index in range(1, self.args.epochs + 1):
                self.execute_training_epoch(epoch_index)
                current_val_loss = self.execute_validation_pass()
                
                if current_val_loss < best_validation_loss:
                    best_validation_loss = current_val_loss
                    if self.is_master:
                        checkpoint_filepath = os.path.join(self.args.checkpoint_dir, "enterprise_model_best.pth")
                        torch.save({
                            'epoch': epoch_index,
                            'state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
                            'best_metric': best_validation_loss,
                            'configuration_arguments': vars(self.args)
                        }, checkpoint_filepath)
                        self.logger.info(f"Persisted new optimal best model checkpoint at epoch {epoch_index} -> {checkpoint_filepath}")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

def main() -> None:
    arguments = parse_args()
    orchestrator = AAPMTrainerOrchestratorEngine(arguments)
    orchestrator.fit()

if __name__ == "__main__":
    main()
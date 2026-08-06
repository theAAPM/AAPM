import os
import time
import json
import logging
import socket
import platform
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, List

class EnterpriseTelemetryDiagnosticRegistry:
    """Central registry for monitoring host diagnostics and environment parameters."""
    @staticmethod
    def capture_system_environment_metadata() -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

class LoggerProxy:
    """Production-grade synchronized file and stream logger with multi-process safeguards."""
    def __init__(self, rank: int, log_dir: str):
        self.rank = rank
        self.logger = logging.getLogger("AAPM_Enterprise_Pipeline_Execution_Telemetry")
        self.logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        
        if rank == 0 and not self.logger.handlers:
            os.makedirs(log_dir, exist_ok=True)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] [PID %(process)d] [Rank %(rank)s]: %(message)s')
            
            fh = logging.FileHandler(os.path.join(log_dir, 'enterprise_execution.log'))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _log_with_rank(self, level: int, msg: str):
        extra = {'rank': str(self.rank)}
        self.logger.log(level, msg, extra=extra)

    def info(self, msg: str):
        self._log_with_rank(logging.INFO, msg)

    def warning(self, msg: str):
        self._log_with_rank(logging.WARNING, msg)

    def error(self, msg: str):
        self._log_with_rank(logging.ERROR, msg)

class MetricTracker:
    """Computes, stores, and synchronizes cross-GPU training scalar metrics across distributed ranks."""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
        
    def sync(self, device: torch.device) -> None:
        if not dist.is_initialized():
            return
        tensor_accumulator = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(tensor_accumulator, op=dist.ReduceOp.SUM)
        self.sum = tensor_accumulator[0].item()
        self.count = tensor_accumulator[1].item()
        self.avg = self.sum / self.count if self.count > 0 else 0.0

class Timer:
    """Context manager for measuring precise execution block latency with automated logging hooks."""
    def __init__(self, name: str, logger: Optional[LoggerProxy] = None):
        self.name = name
        self.logger = logger
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_duration = time.time() - self.start_time
        message = f"Enterprise Operational Timer Block [{self.name}] execution completed in {elapsed_duration:.4f} seconds."
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Inspects, aggregates, and breaks down model parameter allocations across trainable and frozen tiers."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }

def save_json_metadata(path: str, data: Dict[str, Any]) -> None:
    """Serializes experiment arguments, runtime configurations, and performance metrics to persistent JSON storage."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_descriptor:
        json.dump(data, file_descriptor, indent=4, ensure_ascii=False)
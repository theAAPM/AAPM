import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel
from typing import Dict, Tuple, Optional, Any

class DropPath(nn.Module):
    """Stochastic depth regularization layer for dropping residual branches during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_probability = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_probability) * binary_tensor

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization module for stabilizing deep transformers."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)

class WeightInitHelper:
    """Utility helper class executing tailored weight initialization routines."""
    @staticmethod
    def init_weights(module: nn.Module, init_type: str = "trunc_normal") -> None:
        if isinstance(module, nn.Linear):
            if init_type == "trunc_normal":
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network block for modeling sequential video frame dynamics."""
    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding_size = kernel_size // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding_size, groups=hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, temporal_len, hidden_dim = x.shape
        x_transposed = x.transpose(1, 2)
        convolved_tensor = self.conv(x_transposed)
        restored_tensor = convolved_tensor.transpose(1, 2)
        return self.norm(self.act(restored_tensor))

class MultiModalAdapter(nn.Module):
    """Bottleneck adapter network for aligning feature spaces across multi-modal encoders."""
    def __init__(self, in_dim: int, out_dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        mid_dimension = in_dim // bottleneck_ratio
        self.down = nn.Linear(in_dim, mid_dimension)
        self.act = nn.SiLU()
        self.up = nn.Linear(mid_dimension, out_dim)
        self.apply(lambda m: WeightInitHelper.init_weights(m, "xavier"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))

class TextAbsentModule(nn.Module):
    """Baseline ablation module where textual conditioning context is omitted."""
    def __init__(self, hidden_dim: int, drop_path: float = 0.1):
        super().__init__()
        self.temporal_block = TemporalConvNet(hidden_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        return visual_features + self.drop_path(self.temporal_block(visual_features))

class GatedTextVisionFusion(nn.Module):
    """Gated fusion network balancing visual features and pooled text contexts."""
    def __init__(self, hidden_dim: int, drop_path: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size, temporal_len, hidden_dim = visual_features.shape
        text_pooled = text_features.mean(dim=1, keepdim=True).expand(-1, temporal_len, -1)
        concatenated_tensor = torch.cat([visual_features, text_pooled], dim=-1)
        gate_weights = torch.sigmoid(self.gate(concatenated_tensor))
        fusion_output = visual_features * gate_weights + self.proj(text_pooled) * (1.0 - gate_weights)
        return self.norm(visual_features + self.drop_path(fusion_output))

class CrossModalMoEFusion(nn.Module):
    """Mixture-of-Experts routing fusion module for adaptive multi-modal feature integration."""
    def __init__(self, hidden_dim: int, num_experts: int = 4, drop_path: float = 0.1):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), 
                nn.GELU(), 
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_experts)
        ])
        self.drop_path = DropPath(drop_path)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size, temporal_len, hidden_dim = visual_features.shape
        text_context = text_features.mean(dim=1, keepdim=True).expand(-1, temporal_len, -1)
        routing_logits = self.router(text_context)
        routing_probabilities = F.softmax(routing_logits, dim=-1)
        expert_tensor_stack = torch.stack([expert_unit(visual_features) for expert_unit in self.experts], dim=-1)
        moe_aggregated_output = torch.einsum('b t d e, b t e -> b t d', expert_tensor_stack, routing_probabilities)
        return self.norm(visual_features + self.drop_path(moe_aggregated_output))

class TextConstrainModule(nn.Module):
    """Cross-attention text constraint module guiding visual sequence representations."""
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float, drop_path: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.adaptive_attribute_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 128, hidden_dim)) 
        self.drop_path = DropPath(drop_path)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.apply(lambda m: WeightInitHelper.init_weights(m))
        
    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        temporal_length = visual_features.size(1)
        visual_features = visual_features + self.temporal_pos_embed[:, :temporal_length, :]
        attribute_lambda = self.adaptive_attribute_block(text_features) 
        constrained_features, _ = self.cross_attention(
            query=self.norm_q(attribute_lambda),
            key=self.norm_kv(visual_features),
            value=visual_features
        )
        return visual_features + self.drop_path(constrained_features)

class AAPM(nn.Module):
    """Main Adaptive Attribute Prototype Model integrating backbones and ablation fusion variants."""
    def __init__(self, args: Any):
        super().__init__()
        self.args = args
        self.ablation_mode = getattr(args, 'ablation_mode', 'text_constrain')
        
        if "clip" in args.backbone.lower():
            self.backbone = CLIPModel.from_pretrained(args.backbone)
            self.hidden_dim = self.backbone.config.projection_dim
        elif "qwen" in args.backbone.lower():
            self.backbone = AutoModel.from_pretrained(args.backbone, trust_remote_code=True)
            self.hidden_dim = self.backbone.config.hidden_size
            self.adapter = MultiModalAdapter(self.hidden_dim, self.hidden_dim)
        else:
            self.backbone = AutoModel.from_pretrained(args.backbone)
            self.hidden_dim = self.backbone.config.hidden_size

        if args.freeze_backbone:
            self._freeze_backbone()
            
        fusion_registry = {
            "text_absent": TextAbsentModule(self.hidden_dim, args.drop_path_rate),
            "gated_fusion": GatedTextVisionFusion(self.hidden_dim, args.drop_path_rate),
            "moe_fusion": CrossModalMoEFusion(self.hidden_dim, num_experts=4, drop_path=args.drop_path_rate),
            "text_constrain": TextConstrainModule(self.hidden_dim, args.tcm_num_heads, args.tcm_layers, args.dropout, args.drop_path_rate)
        }
        self.fusion_module = fusion_registry.get(self.ablation_mode, fusion_registry["text_constrain"])
        
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            RMSNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        ) if getattr(args, 'use_proj_head', True) else nn.Identity()

        if getattr(args, 'gradient_checkpointing', False) and hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()

    def _freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if "visual_projection" not in name and "text_projection" not in name:
                param.requires_grad = False

    def forward(self, video_frames: torch.Tensor, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, channels, temporal_len, height, width = video_frames.shape
        video_frames_reshaped = video_frames.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * temporal_len, channels, height, width)
        visual_outputs = self.backbone.get_image_features(pixel_values=video_frames_reshaped)
        visual_features = visual_outputs.view(batch_size, temporal_len, self.hidden_dim)
        
        text_features = self.backbone.get_text_features(**text_inputs)
        text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        if hasattr(self, 'adapter'):
            visual_features = self.adapter(visual_features)
            text_features = self.adapter(text_features)
        
        constrained_features = self.fusion_module(text_features, visual_features)
        return self.proj_head(constrained_features)

def compute_dtw_distance(prototype: torch.Tensor, query_feature: torch.Tensor) -> torch.Tensor:
    """Computes exact Dynamic Time Warping alignment distances between prototypes and queries."""
    batch_q, temporal_q, dim_q = query_feature.shape
    batch_p, temporal_p, dim_p = prototype.shape
    distance_matrix = torch.cdist(query_feature.view(batch_q * temporal_q, dim_q), prototype.view(batch_p * temporal_p, dim_p))
    distance_tensor_grid = distance_matrix.view(batch_q, temporal_q, batch_p, temporal_p)
    return distance_tensor_grid.min(dim=-1)[0].mean(dim=-1)

def compute_soft_dtw(prototype: torch.Tensor, query_feature: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Computes differentiable soft Dynamic Time Warping matrix alignment distances."""
    batch_q, temporal_q, dim_q = query_feature.shape
    batch_p, temporal_p, dim_p = prototype.shape
    distance_matrix = torch.cdist(query_feature.view(batch_q * temporal_q, dim_q), prototype.view(batch_p * temporal_p, dim_p))
    distance_matrix_grid = distance_matrix.view(batch_q, temporal_q, batch_p, temporal_p)
    
    recursion_matrix = torch.zeros((batch_q, batch_p, temporal_q + 1, temporal_p + 1), device=query_feature.device, dtype=query_feature.dtype)
    recursion_matrix[:, :, 0, 1:] = float('inf')
    recursion_matrix[:, :, 1:, 0] = float('inf')

    for i in range(1, temporal_q + 1):
        for j in range(1, temporal_p + 1):
            r0 = recursion_matrix[:, :, i - 1, j - 1]
            r1 = recursion_matrix[:, :, i - 1, j]
            r2 = recursion_matrix[:, :, i, j - 1]
            stacked_states = torch.stack([r0, r1, r2], dim=-1)
            soft_minimum = -gamma * torch.logsumexp(-stacked_states / gamma, dim=-1)
            recursion_matrix[:, :, i, j] = distance_matrix_grid[:, :, i - 1, j - 1] + soft_minimum

    return recursion_matrix[:, :, -1, -1].mean(dim=-1)
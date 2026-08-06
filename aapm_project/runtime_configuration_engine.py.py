import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Optional

class EnterpriseRuntimeConfigurationEngine:
    """
    Advanced centralized argument parser and runtime configuration validation engine
    designed for hyperscale distributed multi-modal deep learning clusters.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Enterprise-Grade Distributed Adaptive Attribute Prototype Model (AAPM) Runtime Engine",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._initialize_argument_groups()

    def _initialize_argument_groups(self) -> None:
        # === System & Distributed Parameters ===
        self.parser.add_argument("--seed", type=int, default=42, help="Global pseudo-random generator seed for reproducibility.")
        self.parser.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"], help="Distributed backend communication protocol.")
        self.parser.add_argument("--nccl_timeout", type=int, default=1800, help="NCCL synchronization communication timeout threshold in seconds.")
        self.parser.add_argument("--nccl_debug", type=str, default="WARN", choices=["INFO", "WARN", "DETAIL", "VERSION"], help="Detailed logging level for NCCL diagnostics.")
        self.parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Numerical precision scaling format for automatic mixed precision.")
        self.parser.add_argument("--log_interval", type=int, default=10, help="Batch iteration frequency for printing operational metrics.")
        self.parser.add_argument("--save_freq", type=int, default=5, help="Epoch frequency parameter for persisting model checkpoint weights.")
        self.parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Target filesystem directory for serializing weights.")
        self.parser.add_argument("--log_dir", type=str, default="./logs", help="Directory path designated for tensorboard and telemetry output.")
        self.parser.add_argument("--enable_profiler", action="store_true", help="Flag to activate native PyTorch performance execution profiler.")

        # === Data Loading & Preprocessing Parameters ===
        self.parser.add_argument("--data_source", type=str, default="hdfs", choices=["local", "hdfs", "hive", "s3", "gcs"], help="Underlying enterprise storage interface for dataset retrieval.")
        self.parser.add_argument("--dataset_path", type=str, default="/cluster/data/multi_kinetics", help="Root path or URI string for accessing training shards.")
        self.parser.add_argument("--num_frames", type=int, default=8, help="Temporal clip sampling dimension size.")
        self.parser.add_argument("--resolution", type=int, default=224, help="Spatial target image resolution size.")
        self.parser.add_argument("--num_workers", type=int, default=8, help="Number of worker sub-processes allocated for data loading queues.")
        self.parser.add_argument("--aug_strategy", type=str, default="randaug", choices=["randaug", "trivial", "none", "autoaug"], help="Data augmentation strategy selector.")
        self.parser.add_argument("--color_jitter", type=float, default=0.4, help="Color jitterization intensity coefficient.")

        # === AAM (Attribute Assignment Module) Parameters ===
        self.parser.add_argument("--aam_probs", type=float, nargs="+", default=[1.0, 0.5, 0.5, 0.5], help="Probability weight distributions for attribute assignment variants.")
        self.parser.add_argument("--mask_ratio", type=float, default=0.2, help="Synthetic attribute masking ratio for self-supervised training paths.")

        # === Model & Architecture Parameters ===
        self.parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16", help="Pretrained foundation model identifier or local path.")
        self.parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Boolean flag to freeze base encoder weights during fine-tuning.")
        self.parser.add_argument("--ablation_mode", type=str, default="text_constrain", 
                            choices=["text_absent", "gated_fusion", "moe_fusion", "text_constrain"], help="Architectural ablation configuration selector.")
        self.parser.add_argument("--tcm_num_heads", type=int, default=8, help="Number of attention heads in the text constrain module.")
        self.parser.add_argument("--tcm_layers", type=int, default=1, help="Depth of transformer layers within the attribute attention block.")
        self.parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Stochastic depth drop path probability rate.")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="General hidden layer dropout probability.")
        self.parser.add_argument("--use_proj_head", action="store_true", default=True, help="Toggle utilization of non-linear MLP projection head.")
        self.parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable activation checkpointing to conserve GPU memory footprint.")

        # === Optimization Parameters ===
        self.parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer family selector string.")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate configuration value.")
        self.parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum floor learning rate for cosine annealing schedule.")
        self.parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 regularization weight decay penalty factor.")
        self.parser.add_argument("--batch_size_support", type=int, default=1, help="Support set batch size per data shard.")
        self.parser.add_argument("--batch_size_query", type=int, default=25, help="Query set batch size per data shard.")
        self.parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs to execute.")
        self.parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Proportion of total steps allocated for learning rate warmup.")
        self.parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation frequency steps before optimizer update.")
        self.parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Maximum gradient norm clipping threshold.")
        self.parser.add_argument("--label_smoothing", type=float, default=0.1, help="Cross-entropy label smoothing regularization factor.")
        self.parser.add_argument("--distance_metric", type=str, default="dtw", choices=["dtw", "soft_dtw"], help="Distance measurement metric function.")
        self.parser.add_argument("--temperature", type=float, default=1.0, help="Logits scaling temperature hyperparameter.")
        self.parser.add_argument("--use_ema", action="store_true", default=True, help="Flag to maintain Exponential Moving Average of model weights.")

    def parse(self) -> argparse.Namespace:
        parsed_arguments = self.parser.parse_args()
        self._validate_arguments(parsed_arguments)
        return parsed_arguments

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        if args.lr <= 0.0:
            raise ValueError(f"Invalid learning rate configuration: {args.lr}. Must be strictly positive.")
        if args.epochs <= 0:
            raise ValueError(f"Invalid epochs configuration: {args.epochs}. Must be greater than zero.")
        if args.batch_size_support < 0 or args.batch_size_query < 0:
            raise ValueError("Batch sizes for support and query sets cannot be negative integers.")

def parse_args() -> argparse.Namespace:
    engine = EnterpriseRuntimeConfigurationEngine()
    return engine.parse()
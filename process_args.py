from dataclasses import dataclass, field
from typing import Optional, Tuple
import argparse
import transformers
import os
from datetime import datetime
import shutil


def parser_gen():
    parser = argparse.ArgumentParser(description="Argument parser for model evaluation")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for HuggingFace and PyTorch"
    )

    parser.add_argument(
        "--input_model",
        type=str,
        default="/data/llms/model/meta-llama/Llama-2-7b-hf",
        help="Path to the input model"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./",
        help="Directory to save logs"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (e.g., 0 for cuda:0)"
    )

    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Maximum length of the model for evaluation"
    )

    parser.add_argument(
        "--block_diag",
        type=int,
        default=2,
        help="Split block diagonal into smaller multiples"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type (e.g., float32, float16, bfloat16)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def process_args():
    args, unknown_args = parser_gen()
    args.device = f"cuda:{args.device}"
    args.model_name = os.path.basename(args.input_model)
    return args

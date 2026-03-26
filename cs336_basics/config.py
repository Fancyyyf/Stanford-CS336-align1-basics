from pathlib import Path
import os

# __file__ 是当前脚本的完整路径
# .resolve() 获得绝对路径
# .parent 获得上一级目录
root_dir = Path(__file__).resolve().parent.parent

data_path = root_dir / "data" / "owt_valid.txt"

vocab_size = 10000

DATASET_NAME = "TinyStoriesV2-GPT4-train"

SPECIAL_TOKENS = ["<|endoftext|>"]

MODEL_DIR = "vocab_merge_cache"

VOCAB_PATH = os.path.join(MODEL_DIR, f"vocab_{DATASET_NAME}_{vocab_size}.pkl")

MERGES_PATH = os.path.join(MODEL_DIR, f"merges_{DATASET_NAME}_{vocab_size}.pkl")

OUTPUT_DIR = "encoded_data"
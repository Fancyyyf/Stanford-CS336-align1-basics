from pathlib import Path

# __file__ 是当前脚本的完整路径
# .resolve() 获得绝对路径
# .parent 获得上一级目录
root_dir = Path(__file__).resolve().parent.parent

data_path = root_dir / "data" / "owt_train.txt"

vocab_size = 10000
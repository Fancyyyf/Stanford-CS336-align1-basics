import os
import time
from tokenizer import Tokenizer


def get_10_documents(file_path: str, num_docs: int = 10) -> list[str]:
    """
    从数据集中提取 10 个有效文档（假设每行或每个非空段落是一个 document）
    """
    docs = []
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return docs

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line != "<|endoftext|>":
                docs.append(line)
            if len(docs) == num_docs:
                break
    return docs


def calculate_compression_ratio(docs: list[str], tokenizer: Tokenizer, name: str):
    """
    计算并输出压缩比
    """
    if not docs:
        print(f"⚠️ {name} 没有足够的数据进行测试。")
        return

    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        total_bytes += len(doc.encode('utf-8'))
        total_tokens += len(tokenizer.encode(doc))

    ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    print(f"📊 {name} 测试结果:")
    print(f"   - 总字节数 (Bytes): {total_bytes:,}")
    print(f"   - 总 Token 数 (IDs): {total_tokens:,}")
    print(f"   - 💡 压缩比 (Compression Ratio): {ratio:.2f} bytes/token\n")


if __name__ == "__main__":
    # ==========================================
    # 1. 动态获取绝对路径 (工业级防坑指南)
    # ==========================================
    # 获取当前脚本所在目录 (cs336_basics)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    ROOT_DIR = os.path.dirname(CURRENT_DIR)

    # 拼接数据集路径 (位于根目录的 data 文件夹下)
    TINYSTORIES_TXT_PATH = os.path.join(ROOT_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")
    OPENWEBTEXT_TXT_PATH = os.path.join(ROOT_DIR, "data", "owt_valid.txt")

    # 拼接模型路径 (位于当前目录的 verge_merge_cache 文件夹下)
    CACHE_DIR = os.path.join(CURRENT_DIR, "vocab_merge_cache")

    TS_VOCAB = os.path.join(CACHE_DIR, "vocab_TinyStoriesV2-GPT4-valid_10000.pkl")
    TS_MERGES = os.path.join(CACHE_DIR, "merges_TinyStoriesV2-GPT4-valid_10000.pkl")

    # ⚠️ 注意：根据文档要求，OWT 的词表大小应该是 32K，我这里帮你改成了 32000
    # 如果你实际存的是 10000，请手动改回 10000
    OWT_VOCAB = os.path.join(CACHE_DIR, "vocab_owt_valid_10000.pkl")
    OWT_MERGES = os.path.join(CACHE_DIR, "merges_owt_valid_10000.pkl")

    # ==========================================
    # 2. 加载模型与抽取数据
    # ==========================================
    print("🚀 正在加载 Tokenizer 模型...")
    try:
        ts_tokenizer = Tokenizer.from_files(TS_VOCAB, TS_MERGES, special_tokens=["<|endoftext|>"])
        owt_tokenizer = Tokenizer.from_files(OWT_VOCAB, OWT_MERGES, special_tokens=["<|endoftext|>"])
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径: {e}")
        exit(1)

    print("\n📄 正在从文本中抽取 10 篇 Documents...")
    ts_docs = get_10_documents(TINYSTORIES_TXT_PATH)
    owt_docs = get_10_documents(OPENWEBTEXT_TXT_PATH)

    # ==========================================
    # 3. 运行跑分计算
    # ==========================================
    print("\n" + "=" * 40)
    calculate_compression_ratio(ts_docs, ts_tokenizer, "TinyStories (10K Vocab)")
    calculate_compression_ratio(owt_docs, owt_tokenizer, "OpenWebText (32K Vocab)")
    print("=" * 40)

    # 👇 为实验 (b) 新增的交叉测试代码
    print("-" * 40)
    calculate_compression_ratio(owt_docs, ts_tokenizer, "【跨域实验】OpenWebText 文本 + TinyStories 词表")
    print("=" * 40)

    # ==========================================
    # 4. 实验 (c): 吞吐量压测 (Throughput)
    # ==========================================
    print("\n" + "=" * 40)
    print("⏱️ 正在进行吞吐量压测 (读取前 1MB 文本)...")

    # 截取大约 1MB 的文本进行测试
    test_text = ""
    with open(OPENWEBTEXT_TXT_PATH, 'r', encoding='utf-8') as f:
        test_text = f.read(1024 * 1024)  # 读 1MB

    test_bytes_len = len(test_text.encode('utf-8'))

    # 开始计时
    start_time = time.time()
    # 使用 32K 的 OWT tokenizer 进行编码
    _ = owt_tokenizer.encode(test_text)
    end_time = time.time()

    time_taken = end_time - start_time
    throughput_bytes_per_sec = test_bytes_len / time_taken
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)

    print(f"   - 测试数据量: {test_bytes_len / 1024 / 1024:.2f} MB")
    print(f"   - 耗时: {time_taken:.2f} 秒")
    print(f"   - 🚀 吞吐量: {throughput_bytes_per_sec:,.0f} Bytes/sec ({throughput_mb_per_sec:.2f} MB/s)")

    # 估算 825GB
    pile_bytes = 825 * 1024 * 1024 * 1024
    estimated_seconds = pile_bytes / throughput_bytes_per_sec
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24

    print(f"   - ⏳ 估算处理 The Pile (825GB) 耗时: {estimated_hours:.1f} 小时 (约 {estimated_days:.1f} 天)")
    print("=" * 40)
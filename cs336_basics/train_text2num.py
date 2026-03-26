import numpy as np
import os
import array
import time
import multiprocessing
from typing import Iterator
from tokenizer import Tokenizer

_worker_tokenizer = None


def _init_worker(vocab_path: str, merges_path: str):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])


def _worker_encode(text_chunk: str) -> list[int]:
    return _worker_tokenizer.encode(text_chunk)


def tokenize_and_save_mp(input_txt_path: str, output_npy_path: str, vocab_path: str, merges_path: str,
                         chunk_mb: int = 8):
    print(f"\n🔄 正在处理: {input_txt_path} ...")

    # 如果最终的 .npy 已经存在，直接跳过
    if os.path.exists(output_npy_path):
        print(f"   ⏩ 跳过处理：检测到目标文件已完美生成 ({output_npy_path})")
        print(f"   💾 当前已存文件大小: {os.path.getsize(output_npy_path) / (1024 * 1024):.2f} MB")
        return

    if not os.path.exists(input_txt_path):
        print(f"   ⚠️ 跳过处理：找不到输入文本文件 {input_txt_path}")
        return

    # 定义一个临时二进制文件路径
    temp_bin_path = output_npy_path.replace(".npy", ".temp.bin")

    def safe_chunk_generator() -> Iterator[str]:
        chunk_size_bytes = chunk_mb * 1024 * 1024
        with open(input_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines_buffer = []
            current_bytes = 0
            for line in f:
                lines_buffer.append(line)
                current_bytes += len(line.encode('utf-8'))
                if current_bytes >= chunk_size_bytes:
                    yield "".join(lines_buffer)
                    lines_buffer = []
                    current_bytes = 0
            if lines_buffer:
                yield "".join(lines_buffer)

    total_file_bytes = os.path.getsize(input_txt_path)
    print(f"   📏 文件总大小: {total_file_bytes / (1024 * 1024):.2f} MB")
    print(f"   ⚙️ 正在启动 {multiprocessing.cpu_count()} 核 CPU 全速狂飙...")

    start_time = time.time()
    total_tokens = 0

    # ==========================================
    # 第一阶段：低内存流式写入临时 .bin 文件
    # ==========================================
    with open(temp_bin_path, 'wb') as f_out:
        with multiprocessing.Pool(
                processes=multiprocessing.cpu_count(),
                initializer=_init_worker,
                initargs=(vocab_path, merges_path)
        ) as pool:
            for chunk_idx, ids in enumerate(pool.imap(_worker_encode, safe_chunk_generator())):
                a = array.array('H', ids)
                a.tofile(f_out)
                total_tokens += len(a)
                if (chunk_idx + 1) % 10 == 0:
                    print(f"   🌊 已处理 {((chunk_idx + 1) * chunk_mb):.0f} MB 数据...")

    # ==========================================
    # 第二阶段：零内存消耗转换为标准 .npy 格式
    # ==========================================
    print(f"   📦 正在将临时数据打包为标准的 Numpy 格式...")
    # 使用 memmap 虚拟加载，此时物理内存占用依然为 0
    mmap_array = np.memmap(temp_bin_path, dtype=np.uint16, mode='r')
    # np.save 会通过底层 C 接口直接将硬盘上的 mmap_array 流式转存为 .npy
    np.save(output_npy_path, mmap_array)

    # 功成身退，销毁临时 .bin 文件，节约硬盘空间
    os.remove(temp_bin_path)

    end_time = time.time()
    time_taken = end_time - start_time
    throughput_mb = (total_file_bytes / (1024 * 1024)) / time_taken

    print(f"✅ 保存成功: {output_npy_path} (标准 .npy 格式)")
    print(f"📊 Token 数量: {total_tokens:,}")
    print(f"💾 文件大小: {os.path.getsize(output_npy_path) / (1024 * 1024):.2f} MB")
    print(f"⏱️ 耗时: {time_taken:.2f} 秒")
    print(f"🚀 吞吐量: {throughput_mb:.2f} MB/s\n")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    CACHE_DIR = os.path.join(CURRENT_DIR, "vocab_merge_cache")

    print("=" * 60)
    print("🚀 启动 TinyStories 数据集多进程流水线...")
    print("=" * 60)
    ts_vocab = os.path.join(CACHE_DIR, "vocab_TinyStoriesV2-GPT4-train_10000.pkl")
    ts_merges = os.path.join(CACHE_DIR, "merges_TinyStoriesV2-GPT4-train_10000.pkl")

    tokenize_and_save_mp(os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt"), os.path.join(DATA_DIR, "ts_train.npy"),
                         ts_vocab, ts_merges, chunk_mb=8)
    tokenize_and_save_mp(os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt"), os.path.join(DATA_DIR, "ts_valid.npy"),
                         ts_vocab, ts_merges, chunk_mb=8)

    print("=" * 60)
    print("🚀 启动 OpenWebText 数据集多进程流水线...")
    print("=" * 60)
    owt_vocab = os.path.join(CACHE_DIR, "vocab_owt_train_10000.pkl")
    owt_merges = os.path.join(CACHE_DIR, "merges_owt_train_10000.pkl")

    tokenize_and_save_mp(os.path.join(DATA_DIR, "owt_train.txt"), os.path.join(DATA_DIR, "owt_train.npy"), owt_vocab,
                         owt_merges, chunk_mb=8)
    tokenize_and_save_mp(os.path.join(DATA_DIR, "owt_valid.txt"), os.path.join(DATA_DIR, "owt_valid.npy"), owt_vocab,
                         owt_merges, chunk_mb=8)
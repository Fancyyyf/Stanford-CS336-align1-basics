from cs336_basics import config
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import copy
import os
import pickle
import multiprocessing
import collections
import time
import math
from functools import partial

'''
注：笔者犯傻了为了显示多写了b2i,u2b函数实际对训练没有必要，正常能训练就够了。实际只是方便调试输出的时候方便显示。

注：记得更新多线程的并行，对于大数据量只占一个线程计算频率太慢
'''


# 正则化工具
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# 词表可见性映射
def get_byte_unicode_mapping():
    """
    生成 256 个字节到 Unicode 字符的双向映射字典。
    返回:
        b2u (dict[int, str]): 字节值 (0-255) 到 Unicode 字符的映射
        u2b (dict[str, int]): Unicode 字符到字节值的反向映射
    """
    # 已经是“可见字符”的字节
    # 包括标准 ASCII 可打印字符，以及一部分拉丁文扩展字符
    bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
    )

    # 复制一份作为对应的 Unicode 字符列表
    cs = bs[:]

    # 遍历“不可见/控制”字节 (比如 0, 10, 255 等)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            # 映射到到 Unicode 码点 256 之后的可见字符上
            cs.append(256 + n)
            n += 1

    # 将整数码点转换为真正的 Python 单字符字符串
    cs = [chr(n) for n in cs]

    # 3. 组合成正向字典 (Byte -> Unicode)
    b2u = dict(zip(bs, cs))

    # 4. 组合成反向字典 (Unicode -> Byte)
    u2b = {v: k for k, v in b2u.items()}

    return b2u, u2b

def process_chunk_by_offsets(file_path: str, start: int, end: int, b2u: dict, special_tokens: list[str]) -> collections.Counter:
    """
    Worker 函数：处理单块文本，提取预分词并统计频率。

    参数:
        text_chunk: 分配给当前核心处理的文本块字符串。
        b2u: 已经生成好的 byte_to_unicode 映射字典 (dict[int, str])。
        special_token: 需要保护和剥离的特殊文档分隔符。

    返回:
        collections.Counter: 当前文本块的预分词词频统计。
    """
    chunk_counts = collections.Counter()

    # 子进程自己打开文件，定位并读取数据
    # 注意：在处理 seek 和 read 字节偏移时，最好以二进制模式 ('rb') 读取，
    # 读出来之后再用 utf-8 解码为文本，以避免变长字符导致的截断问题。
    with open(file_path, 'rb') as f:
        f.seek(start)
        # 读取指定长度的字节并解码为字符串
        raw_bytes = f.read(end - start)
        text_chunk = raw_bytes.decode('utf-8', errors='ignore')  # 忽略边界可能切破的一两个多字节字符碎片

    # 针对文档分隔符进行分割
    # 正则表达式的 re.escape 可以自动处理特殊字符，确保它们被当作字面量匹配
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        parts = re.split(split_pattern, text_chunk)
    else:
        parts = [text_chunk]

    for part in parts:
        # 跳过连续特殊 token 造成的空字符串
        if not part:
            continue

        # 正则化匹配
        for match in PAT.finditer(part):
            token_str = match.group()  # 拿出正则化后的字符串

            # 编码转化为字节
            b_text = token_str.encode("utf-8")

            # b2u 映射：将纯字节转化为可见的 Unicode 字符元组
            mapped_word = tuple(b2u[b] for b in b_text)

            chunk_counts[mapped_word] += 1

    return chunk_counts


def _worker_wrapper(args, b2u, special_tokens):
    """
    拆包代理函数：
    进程池扔过来的 args 是一个完整的元组 ("file.txt", 0, 5000)
    这个函数负责把元组拆开，然后再精准地喂给真正的处理函数。
    """
    file_path, start, end = args
    return process_chunk_by_offsets(file_path, start, end, b2u, special_tokens)

# 预分词并行
def get_pretokenized_counts(file_path: str, b2u: dict, special_tokens: list[str] = None,
                            num_workers: int = 8) -> collections.Counter:

    output_dir = "pretoken_cache"
    # exist_ok=True 表示如果文件夹已存在，不会报错
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)# 提取文件名

    # 使用 os.path.join 拼接路径，这能自动处理不同系统的斜杠（/ 或 \）
    cache_path = os.path.join(output_dir, f"cached_counts_{base_name}.pkl")


    if os.path.exists(cache_path):
        print(f"检测到本地缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    global_counts = collections.Counter()

    # 利用pretokenization的函数进行切块
    if special_tokens:
        special_token_bytes = special_tokens[0].encode('utf-8')
    else:
        # 如果连 special_tokens 都没有，随便给个默认值防止 find_chunk_boundaries 报错
        special_token_bytes = b"<|endoftext|>"


    # 动态控制 Chunk 物理大小 (防止撑爆单进程)
    # 强制将每块大小压榨到 32MB 左右
    file_size = os.path.getsize(file_path)
    chunk_size = 32 * 1024 * 1024  # 强制规定每块约 32MB
    # 确保至少有 num_workers * 4 块，但遇到 11GB 文件时会切成约 350 块
    desired_chunks = max(num_workers * 4, math.ceil(file_size / chunk_size))

    print(f"文件大小: {file_size / (1024 ** 3):.2f} GB，将被切分为约 {desired_chunks} 块进行极速处理...")

    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=desired_chunks,
                                               split_special_token=special_token_bytes)

    # 将边界点转换成给子进程的坐标任务包
    chunk_offsets = []
    for i in range(len(boundaries) - 1):
        chunk_offsets.append((file_path, boundaries[i], boundaries[i + 1]))

    # 固定全局参数
    # 把“拆解包函数”扔给进程池
    worker = partial(_worker_wrapper, b2u=b2u, special_tokens=special_tokens)

    # 开启进程池
    # 进程池内存强制回收 (maxtasksperchild) 可以防止某些内存泄漏的库在长时间运行后占满内存，尤其是在处理大型文本时。
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for chunk_counts in pool.imap_unordered(worker, chunk_offsets): # 乱序返回结果，减少等待时间
            global_counts.update(chunk_counts)

    print(f"预处理完成！正在存入本地缓存...")
    with open(cache_path, "wb") as f:
        pickle.dump(global_counts, f)  # 序列化原结构保存

    return global_counts


# 统计词表对的出现
def get_initial_stats(word_counts: dict) -> tuple[dict, collections.defaultdict]:
    """
    只运行一次遍历词表，建立正向的频率对 (pair_counts) 和反向索引 (pair_to_words)。
    """
    pair_counts = collections.defaultdict(int)
    # 反向索引：记录某个相邻对存活在哪些“单词元组”中
    pair_to_words = collections.defaultdict(set)

    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    return pair_counts, pair_to_words

# 给定一个单词元组和目标组合，左到右贪心匹配
def merge_word(word: tuple, pair: tuple) -> tuple:
    new_word = []
    i = 0
    while i < len(word):
        # 如果还没到结尾，且当前字符和下一个字符刚好是我们想合并的 pair
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2  # 一次性跳过两个位置
        else:
            # 没匹配上，原样保留
            new_word.append(word[i])
            i += 1

    return tuple(new_word)

# BPE 训练主循环：执行动态账本更新，产出工业级词表和合并规则。
def train_bpe(word_counts: dict, b2u: dict, vocab_size: int, special_tokens: list[str] = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = []

    # 生成反向字典,用于最终返回时的字节还原
    u2b = {v: k for k, v in b2u.items()}

    # 基础词表 (0~255) 和空的合并规则
    vocab = {idx: bytes([u2b[char]]) for idx, char in b2u.items()}
    merges = []

    next_vocab_id = 256
    for st in special_tokens:
        vocab[next_vocab_id] = st.encode("utf-8")
        next_vocab_id += 1

    # 将 word_counts 的键从 Unicode 字符元组转换为对应的 bytes 元组，以便后续处理
    # 也就是空格等难显示的字符向后映射的还原
    bytes_word_counts = {}
    for word_tuple, freq in word_counts.items():
        bytes_word = tuple(bytes([u2b[char]]) for char in word_tuple)
        bytes_word_counts[bytes_word] = freq

    # 调用辅助函数一次遍历，获得有序对的频率与pair_to_words包含关系
    pair_counts, pair_to_words = get_initial_stats(bytes_word_counts)

    num_merges = vocab_size - 256 - len(special_tokens)  # 还需要合并多少次

    # 进行合并循环
    for step in range(num_merges):
        # 如果词对空了（所有词都只剩一个整体了），提前结束
        if not pair_counts:
            break

        # 寻找最高频词对
        # 如果频率相同，使用字典序进行优先级排序，x[1]代表频率, x[0]代表字典序
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # 通过反向索引，查找需要修改的单词集合
        # 注意要用 list() 复制一份，因为我们接下来的操作会修改原字典
        words_to_process = list(pair_to_words[best_pair])

        for word in words_to_process:
            freq = bytes_word_counts[word]

            # 拿到合并后的新单词
            new_word = merge_word(word, best_pair)

            # 旧pair清理
            for i in range(len(word) - 1):
                old_pair = (word[i], word[i + 1])
                pair_counts[old_pair] -= freq    # 除去老的pair_count
                if pair_counts[old_pair] <= 0:   # 防止直接删除导致的 KeyError
                    del pair_counts[old_pair]
                # 用 discard 而不是 remove，防止同一个词出现重叠对时的报错
                pair_to_words[old_pair].discard(word)

            # 新pair更新
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pair_counts[new_pair] += freq    # 由于collections.defaultdict(int)的统计特性，这里不需要担心 new_pair 不存在的情况
                pair_to_words[new_pair].add(new_word)

            # 更新全局词表
            del bytes_word_counts[word]
            bytes_word_counts[new_word] = freq

        p1_bytes = best_pair[0]
        p2_bytes = best_pair[1]

        # 记录到 merges
        merges.append((p1_bytes, p2_bytes))

        # 记录到 vocab (分配新的 Token ID)
        # 合并字节流
        vocab[next_vocab_id] = p1_bytes + p2_bytes
        next_vocab_id += 1

        # 彻底清理已经被合并干掉的 best_pair
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

        # 每合并 1000 次打印一次进度
        if (step + 1) % 1000 == 0:
            print(f"进度: {step + 1}/{num_merges} 次合并完成. 当前新词: {vocab[next_vocab_id - 1]}")

    return vocab, merges


def bpe_workshop(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 动态生成缓存文件名，确保不同输入文件和词表大小的训练结果不会冲突
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = "vocab_merge_cache"
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, f"vocab_{base_name}_{vocab_size}.pkl")
    merges_path = os.path.join(output_dir, f"merges_{base_name}_{vocab_size}.pkl")

    # 拦截检查，实现秒级加载
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"✅ 检测到本地已存在训练好的模型 [{base_name} | size: {vocab_size}]，直接加载...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return vocab, merges

    print(f"🚀 未找到完整模型缓存，开始走完整训练流水线 [{base_name} | size: {vocab_size}]...")

    # 获取基础映射字典
    b2u, u2b = get_byte_unicode_mapping()

    # 分词耗时统计
    print("\n⏳ 开始执行多进程预分词 (Pre-tokenization)...")
    start_pretoken = time.perf_counter()

    counts = get_pretokenized_counts(input_path, b2u=b2u, special_tokens=special_tokens)

    pretoken_duration = time.perf_counter() - start_pretoken
    print(f"⏱️ 预分词彻底完成！耗时: {pretoken_duration:.4f} 秒")

    counts_for_training = copy.deepcopy(counts)  # 拷贝一份

    # 执行核心训练，拿到交付物
    vocab, merges = train_bpe(counts_for_training, b2u, vocab_size, special_tokens)

    # 训练完成后保存模型文件
    print(f"💾 训练完成，正在保存模型文件至 {output_dir}/ ...")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


# 测试一下
if __name__ == "__main__":
    start_total = time.perf_counter()

    test_file = config.data_path
    special_tok = ["<|endoftext|>"]
    target_vocab_size = config.vocab_size

    # 自动生成包含特殊 Token 的测试文件 (模拟语料)
    if not os.path.exists(test_file):
        print("正在生成测试文件...")
        with open(test_file, "w", encoding="utf-8") as f:
            for _ in range(50000):
                # 【修复】special_tok 是列表，写入文件时需要取 [0]
                f.write(
                    f"Hello world! This is a test story.{special_tok[0]}Another story begins here! Stanford CS336 is awesome.{special_tok[0]}"
                )

    # 启动测试
    print(f"\n🚀 开始 BPE Workshop 调度！目标词表大小: {target_vocab_size}")
    start_workshop = time.perf_counter()

    # 内部自动处理：检测缓存 -> 预处理 -> 拷贝词频 -> 核心训练 -> 存盘保存
    vocab, merges = bpe_workshop(test_file, target_vocab_size, special_tok)

    workshop_duration = time.perf_counter() - start_workshop
    print(f"\n🏆 BPE Workshop 执行完成！(含 I/O 调度) 耗时: {workshop_duration:.4f} 秒")

    # 统计分析
    start_stats = time.perf_counter()

    # 统计词表中最长的前 5 个词元 (基于字节长度)
    # vocab.values() 存储的直接就是 bytes 类型，直接按 len 排序即可
    longest_tokens = sorted(vocab.values(), key=len, reverse=True)[:5]

    stats_duration = time.perf_counter() - start_stats

    print("\n📊 训练后统计数据:")
    print(f"✅ 最终词表大小: {len(vocab)}")
    print(f"✅ 最终合并规则: {len(merges)} 条")

    print("\n🔍 长度最长的前 5 个词元 (用于检验长单词学习效果):")
    for token_bytes in longest_tokens:
        # 因为已经是底层 bytes，直接 decode 即可，不用再查 u2b 字典！
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"   - 可读形态: {token_str!r:<20} | 长度: {len(token_bytes)} 字节 | 底层形态: {str(token_bytes)}")

    print(f"\n⏱️ 统计分析耗时: {stats_duration:.4f} 秒")
    print(f"✨ 总运行耗时: {time.perf_counter() - start_total:.4f} 秒")
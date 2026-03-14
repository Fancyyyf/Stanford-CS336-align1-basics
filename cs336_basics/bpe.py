import config
from pretokenization_example import find_chunk_boundaries
import regex
import collections
import os
import pickle
import multiprocessing
import collections
from functools import partial


# 正则化工具
PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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

def process_chunk_by_offsets(file_path: str, start: int, end: int, b2u: dict, special_token: str = "<|endoftext|>") -> collections.Counter:
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
    parts = text_chunk.split(special_token)

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


def _worker_wrapper(args, b2u, special_token):
    """
    拆包代理函数：
    进程池扔过来的 args 是一个完整的元组 ("file.txt", 0, 5000)
    这个函数负责把元组拆开，然后再精准地喂给真正的处理函数。
    """
    file_path, start, end = args
    return process_chunk_by_offsets(file_path, start, end, b2u, special_token)


def get_pretokenized_counts(file_path: str, b2u: dict, special_token: str = "<|endoftext|>",
                            num_workers: int = 8) -> collections.Counter:

    output_dir = "cache"
    # exist_ok=True 表示如果文件夹已存在，不会报错
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(file_path)# 提取文件名

    # 使用 os.path.join 拼接路径，这能自动处理不同系统的斜杠（/ 或 \）
    cache_path = os.path.join(output_dir, f"cached_counts_{base_name}.pkl")


    if os.path.exists(cache_path):
        print(f"检测到本地缓存: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"开始多进程预处理...")
    global_counts = collections.Counter()

    # 利用pretokenization的函数进行切块
    special_token_bytes = special_token.encode('utf-8')
    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_workers*2, split_special_token=special_token_bytes)

    # 将边界点转换成给子进程的坐标任务包
    chunk_offsets = []
    for i in range(len(boundaries) - 1):
        chunk_offsets.append((file_path, boundaries[i], boundaries[i + 1]))

    # 固定全局参数
    # 把“拆解包函数”扔给进程池
    worker = partial(_worker_wrapper, b2u=b2u, special_token=special_token)

    # 开启进程池
    with multiprocessing.Pool(processes=num_workers) as pool:
        for chunk_counts in pool.imap_unordered(worker, chunk_offsets): # 乱序返回结果，减少等待时间
            global_counts.update(chunk_counts)

    print(f"预处理完成！正在存入本地缓存...")
    with open(cache_path, "wb") as f:
        pickle.dump(global_counts, f)  # 序列化原结构保存

    return global_counts



# 测试一下
if __name__ == "__main__":
    test_file = config.data_path
    special_tok = "<|endoftext|>"

    # 自动生成一个包含特殊 Token 的测试文件 (模拟几兆的语料)
    if not os.path.exists(test_file):
        print("正在生成测试文件...")
        with open(test_file, "w", encoding="utf-8") as f:
            for _ in range(50000):
                f.write(
                    f"Hello world! This is a test story.{special_tok}Another story begins here! Stanford CS336 is awesome.{special_tok}")

    # 获取基础映射字典
    b2u, u2b = get_byte_unicode_mapping()

    # 运行多进程预处理管线
    counts = get_pretokenized_counts(test_file, b2u=b2u, special_token=special_tok)

    # 打印结果验证
    print("\n🎉 最终统计结果 (前 5 个最高频词汇:")

    for word_tuple, freq in counts.most_common(5):
        # 1. 还原：用 u2b 反向字典，把每一个 BPE 字符映射回底层的 0-255 字节值
        raw_bytes = bytes([u2b[c] for c in word_tuple])

        # 2. 解码：将纯正的字节流用 UTF-8 解码为人类可读的字符串
        # 加上 errors='replace' 是为了防止某些在截断边缘损坏的多字节字符引发报错
        readable_word = raw_bytes.decode('utf-8', errors='replace')

        # 顺便保留它在 BPE 字典里拼接后的“内部形态”，方便你对比
        bpe_internal_form = "".join(word_tuple)

        print(f"  可读形态: {readable_word:<15} | 字典底层: {bpe_internal_form:<15} | 频率: {freq}")

import os
import pickle
import regex as re
from typing import Iterable, Iterator
import array
from cs336_basics import config


PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        初始化 Tokenizer，构建查表字典与优先级字典。
        """
        self.vocab = vocab.copy()  # 浅拷贝，防止污染外部传入的字典
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # 确保 special_tokens 都在 vocab 中，如果不在则追加
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.vocab.values():
                # 分配当前词表中最大的 ID + 1
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = st_bytes

        # 反向词表：用于 encode 最后一步的查表 (bytes -> ID)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # 构建 O(1) 查找的优先级字典
        # 字典的 Value 就是该 pair 在 merges 列表中的索引（越小代表优先级越高）
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将文本字符串编码为整数 ID 列表。
        """
        # 切分特殊 Token
        if self.special_tokens:
            # 使用捕获组 ()，这样 re.split 在切分时会保留分隔符（即特殊 Token 本身）
            # 例如 "A<|end|>B" 会被切成 ['A', '<|end|>', 'B']
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)    # 先按长度排序，确保长的特殊 Token 优先匹配

            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if not part:
                continue

            # 如果是特殊的 Token，直接查表拿 ID，跳过正则和 BPE 合并
            if part in self.special_tokens:
                ids.append(self.inverse_vocab[part.encode("utf-8")])
                continue

            # 对普通文本进行正则切分
            for match in PAT.finditer(part):
                chunk = match.group().encode("utf-8")
                # 合并循环
                merged_bytes = self._merge_chunk(chunk)

                # 映射为整数 ID 并追加
                ids.extend(self.inverse_vocab[b] for b in merged_bytes)

        return ids


    def _merge_chunk(self, chunk_bytes: bytes) -> list[bytes]:
        """
        核心辅助函数：执行单块的 BPE 微观合并循环
        """
        # 打碎成最基础的单字节列表
        b_list = [bytes([b]) for b in chunk_bytes]

        while len(b_list) >= 2:
            # 寻找当前列表中优先级最高的相邻 pair
            best_pair = None
            best_rank = float('inf')

            for i in range(len(b_list) - 1):
                pair = (b_list[i], b_list[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 如果列表里没有任何 pair 在规则库中，说明合并彻底结束
            if best_pair is None:
                break

            # 执行合并替换
            new_b_list = []
            i = 0
            while i < len(b_list):
                if i < len(b_list) - 1 and b_list[i] == best_pair[0] and b_list[i + 1] == best_pair[1]:
                    new_b_list.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_b_list.append(b_list[i])
                    i += 1
            b_list = new_b_list

        return b_list


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码器：利用 Python 生成器惰性求值，处理海量数据。
        """
        for text_chunk in iterable:
            # yield from 等价于遍历 encode 返回的列表并逐个 yield
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        """
        将整数 ID 列表还原为人类可读文本。
        """
        # 将所有的 ID 还原为 bytes，并拼接成一条完整的字节流
        # 如果遇到词表中不存在的异常 ID，默认返回空字节以防崩溃
        raw_bytes = b"".join(self.vocab.get(idx, b"") for idx in ids)

        # 严格遵守文档要求：遇到无法解析的破碎多字节字符，静默替换为 U+FFFD ()
        return raw_bytes.decode("utf-8", errors="replace")


# 独立函数，不属于 Tokenizer 类，专门负责将文本文件流式编码并保存为二进制文件
def tokenize_and_save_dataset(
        input_path: str,
        output_path: str,
        tokenizer,
        chunk_size: int = 1024 * 1024  # 每次只读 1MB 文本进内存
):
    """
    利用 Tokenizer 的流式能力，将超大文本文件编码并落盘为高效的二进制文件 (.bin)。
    无论输入文件是 11GB 还是 1000GB，内存占用始终在 10MB 以内。
    """

    # ==========================================
    # 🕵️ 阶段一：前端读取器 (按块读取，防止单行过长撑爆内存)
    # ==========================================
    def text_chunk_generator() -> Iterator[str]:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                text = f.read(chunk_size)
                if not text:
                    break
                yield text  # 惰性吐出 1MB 的文本

    print(f"🚀 开始启动流式编码流水线...")
    print(f"📥 输入源: {input_path}")
    print(f"💾 输出至: {output_path}")

    # ==========================================
    # ⚙️ 阶段二：接驳核心管道 (魔法发生的地方)
    # ==========================================
    # 将前端读取器直接喂给 Tokenizer，拿到一个无尽的 ID 吐出流
    token_id_iterator = tokenizer.encode_iterable(text_chunk_generator())

    # ==========================================
    # 🪣 阶段三：后端写入器 (缓冲池 + 二进制冲刷)
    # ==========================================
    # 'i' 代表 4 字节有符号整数 (int32)，非常适合保存 Token ID
    buffer = array.array('i')
    flush_limit = 1000000  # 蓄水池：攒够 100 万个 ID 就落盘一次

    total_tokens = 0

    with open(output_path, 'wb') as f_out:
        for token_id in token_id_iterator:
            buffer.append(token_id)
            total_tokens += 1

            # 当蓄水池满了，一口气冲刷进硬盘，然后清空蓄水池
            if len(buffer) >= flush_limit:
                buffer.tofile(f_out)
                buffer = array.array('i')  # 重新换个空桶
                print(f"   🌊 已落盘 {total_tokens:,} 个 Tokens...")

        # 收尾：处理最后半桶水（没满 100 万的残余数据）
        if buffer:
            buffer.tofile(f_out)
            print(f"   🌊 已落盘 {total_tokens:,} 个 Tokens...")

    print(f"✅ 彻底完工！总计产生 {total_tokens:,} 个 Tokens。")
    print(f"📊 二进制文件大小约: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")



# 本地测试
if __name__ == "__main__":

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)  # 自动创建存放 bin 文件的文件夹

    print("=" * 60)
    print("🤖 工业级 Byte-Level BPE Tokenizer 控制台")
    print("=" * 60)

    if not os.path.exists(config.VOCAB_PATH) or not os.path.exists(config.MERGES_PATH):
        print(f"❌ 严重错误：未在 {config.MODEL_DIR}/ 目录下找到模型文件！")
        print(f"   预期寻找：\n   - {config.VOCAB_PATH}\n   - {config.MERGES_PATH}")
        print("   请确认之前是否已成功运行 bpe_workshop 并保存了模型。")
        exit(1)

    print(f"⏳ 正在从 {config.MODEL_DIR}/ 极速加载词表和合并规则...")
    # 完美利用 from_files 工厂方法一键拉起！
    tokenizer = Tokenizer.from_files(config.VOCAB_PATH, config.MERGES_PATH, special_tokens=config.SPECIAL_TOKENS)
    print(f"✅ 挂载成功！当前词表大小: {len(tokenizer.vocab)}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 💬 2. 交互式主循环 (REPL)
    # ---------------------------------------------------------
    print("💡 模式指引:")
    print("   1. 直接输入文本：测试单句 Encode/Decode 效果。")
    print("   2. 输入 'file:文件路径'：启动工业级流水线，将大文件转化为 .bin 存盘。")
    print("   3. 输入 'q' 或 'quit'：退出控制台。")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n>>> 请输入指令或文本: ").strip()
        except (KeyboardInterrupt, EOFError):
            # 优雅处理 Ctrl+C 或 Ctrl+D 退出
            print("\n👋 退出 Tokenizer 控制台，再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ['q', 'quit', 'exit']:
            print("👋 退出 Tokenizer 控制台，再见！")
            break

        # -----------------------------------------------------
        # 分支 A：处理大文件转码请求 (格式: file:data/xxx.txt)
        # -----------------------------------------------------
        if user_input.startswith("file:"):
            target_file = user_input[5:].strip()

            if not os.path.exists(target_file):
                print(f"❌ 错误：找不到文件 '{target_file}'")
                continue

            # 动态生成输出文件名：encoded_文本名_词表大小.bin
            base_name = os.path.splitext(os.path.basename(target_file))[0]
            output_file = os.path.join(config.OUTPUT_DIR, f"encoded_{base_name}_{config.vocab_size}.bin")

            # 调用流水线
            tokenize_and_save_dataset(target_file, output_file, tokenizer)

        # -----------------------------------------------------
        # 分支 B：处理普通文本的单句测试
        # -----------------------------------------------------
        else:
            print(f"🔍 原始文本: {user_input!r}")

            # 1. 编码
            encoded_ids = tokenizer.encode(user_input)
            print(f"🧩 编码结果: {encoded_ids}")
            print(f"📏 Token 数量: {len(encoded_ids)}")

            # 2. 解码验证
            decoded_text = tokenizer.decode(encoded_ids)
            print(f"🔄 解码还原: {decoded_text!r}")

            # 3. 严格一致性校验
            if decoded_text == user_input:
                print("✅ 校验通过：Decode 文本与原始输入绝对一致！")
            else:
                print("⚠️ 校验警告：还原文本与原文本存在差异（可能是未闭合的奇怪字符导致）")
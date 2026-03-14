# Q1
![img.png](img.png)
1. 返回为空字符（U+0000）
2. \x00
3. 正常拼接

# Q2
![img_1.png](img_1.png)
### (a) 为什么选择 UTF-8 而不是 UTF-16 或 UTF-32？

**Deliverable (One-to-two sentence response):** UTF-8 is preferred because it uses a variable-length encoding that represents common ASCII characters with a single byte, avoiding the excessive padding and sequence bloat that wider encodings like UTF-16 or UTF-32 would introduce. This keeps the input sequence lengths manageable and memory-efficient, which is crucial for training language models.

**（中文原理解析）**：UTF-8 是互联网的主流编码（占比超过 98%）。在处理绝大多数英文文本时，UTF-8 每个字母只占 1 个字节。如果换成 UTF-32，每个字母都要占 4 个字节，比如字母 "a" 会变成 `\x00\x00\x00\x61`。这些大量的冗余空字节会导致序列变得极其冗长，拖慢模型的训练速度并增加计算成本 。

### (b) 为什么 `decode_utf8_bytes_to_str_wrong` 这个函数是错的？

**Deliverable (Example & one-sentence explanation):** Example input: `b'\xe3\x81\x93'` (the UTF-8 encoding for the Japanese character 'こ'). Explanation: The function is incorrect because it attempts to decode every single byte individually, which fails to recognize that in UTF-8, a single Unicode character can span multiple bytes.

**（中文原理解析）**：就像我们之前聊过的，UTF-8 里一个字符不一定只占一个字节 。比如汉字或者日文往往占 3 个字节 。这段错误代码强行把这 3 个字节拆散了，一次只丢给解码器 1 个字节。解码器看到残缺的 `\xe3`，根本不知道它是什么字符，自然会直接抛出 `UnicodeDecodeError` 报错。

### (c) 给出一个无法解码为任何 Unicode 字符的双字节序列。

**Deliverable (Example & one-sentence explanation):**

Example: `b'\xff\xff'`.

Explanation: This sequence does not decode to any character because the byte `\xff` is entirely illegal in the UTF-8 encoding standard, violating its strict bit-pattern rules for valid sequences.

**（中文原理解析）**：UTF-8 是一套非常严格的“拼图游戏”，比如多字节字符的首字节必须以 `110` 或 `1110` 开头，后续字节必须以 `10` 开头。像 `\xff`（二进制是 `11111111`）这种字节在 UTF-8 的规则本里是不存在的，所以哪怕你传给解码器，它也只能宣告失败。














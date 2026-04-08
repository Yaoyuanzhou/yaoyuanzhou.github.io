"""
BBPE（字节级 BPE）完整实现
======================================
GPT-2、GPT-3/4、LLaMA 3、Qwen 等使用的字节级分词算法。

与普通 BPE 的唯一区别：
  - 普通 BPE：初始词表 = Unicode 字符（可能有 10 万+种）
  - BBPE：初始词表 = 256 个字节（0-255），任何文本都能表示，绝无 OOV

背景知识：
  UTF-8 编码规则
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ Unicode 范围  │ UTF-8 字节数                                      │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ U+0000–007F  │ 1 字节（ASCII，如英文字母）                         │
  │ U+0080–07FF  │ 2 字节（拉丁扩展、阿拉伯文等）                       │
  │ U+0800–FFFF  │ 3 字节（基本多文种平面，含大部分汉字）                  │
  │ U+10000+     │ 4 字节（emoji、生僻字、扩展汉字等）                    │
  └──────────────┴──────────────────────────────────────────────────┘

  所以：
    'A'  → [65]              （1 个字节）
    '你' → [228, 189, 160]   （3 个字节）
    '𠀀' → [240, 160, 128, 128]（4 个字节，生僻字）
    '🌍' → [240, 159, 140, 141]（4 个字节，emoji）

  BBPE 先把所有文本转为字节序列，再在字节上做 BPE 合并。

核心论文：
  - Wang et al. 2020, "Neural Machine Translation with Byte-Level Subwords"
  - GPT-2 技术报告（Radford et al. 2019）
"""

from collections import Counter
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════
# STEP 1：文本 ↔ 字节序列 转换工具
# ═══════════════════════════════════════════════════════════

def text_to_bytes(text: str) -> List[int]:
    """
    把文本编码为 UTF-8 字节列表。

    每个字节是 0-255 的整数，这就是 BBPE 的"字母表"。

    例子：
        'A'  → [65]
        '你好' → [228, 189, 160, 229, 165, 189]
        '🌍' → [240, 159, 140, 141]

    Args:
        text: 任意 Unicode 文本

    Returns:
        UTF-8 字节整数列表
    """
    return list(text.encode('utf-8'))


def bytes_to_text(byte_list: List[int]) -> str:
    """
    把字节列表解码为文本。

    如果字节序列是合法的 UTF-8，返回对应文本；
    否则用 replace 模式跳过非法字节（避免崩溃）。

    Args:
        byte_list: 字节整数列表（每个元素 0-255）

    Returns:
        解码后的文本
    """
    return bytes(byte_list).decode('utf-8', errors='replace')


def byte_to_readable(b: int) -> str:
    """
    把字节 id（0-255）转为可打印的字符串表示。

    GPT-2 的实现里，把控制字符映射到特殊 Unicode 字符，
    这里简化为直接用 b<数字> 表示（如 b0, b65, b228）。

    Args:
        b: 字节值（0-255）

    Returns:
        可读字符串
    """
    if 32 <= b < 127:
        # 可打印 ASCII 字符直接用
        return chr(b)
    else:
        # 不可打印字节用 b<数字> 表示
        return f'<b{b}>'


# ═══════════════════════════════════════════════════════════
# STEP 2：字节级词频词典
# ═══════════════════════════════════════════════════════════

def get_byte_vocab(corpus: List[str]) -> Dict[Tuple[int, ...], int]:
    """
    将语料转换为字节级词频词典。

    与普通 BPE 不同：
        - 普通 BPE 的 key 是字符序列字符串，如 "l o w </w>"
        - BBPE 的 key 是字节元组，如 (108, 111, 119)（"low" 的 UTF-8 字节）

    词尾标记：BBPE 通常直接用空格字节（32）而非 </w> 来区分词边界。
    或者更常见的是：词首加特殊标记（GPT-2 用 Ġ 表示前导空格）。
    这里简化处理：每个词的字节直接作为 key，通过空格分词。

    Args:
        corpus: 文本语料列表

    Returns:
        {字节元组: 出现次数}
    """
    vocab: Dict[Tuple[int, ...], int] = Counter()
    for text in corpus:
        for word in text.strip().split():
            byte_seq = tuple(text_to_bytes(word))
            vocab[byte_seq] += 1
    return dict(vocab)


# ═══════════════════════════════════════════════════════════
# STEP 3：统计字节 bigram 频率
# ═══════════════════════════════════════════════════════════

def get_byte_stats(vocab: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
    统计所有词中相邻字节对的频率。

    注意：这里的 key 是字节整数对 (int, int)，
    而普通 BPE 中是字符串对 (str, str)。

    Args:
        vocab: 字节级词频词典

    Returns:
        {(byte_a, byte_b): 频率}
    """
    pairs: Dict[Tuple[int, int], int] = Counter()
    for byte_seq, freq in vocab.items():
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += freq
    return dict(pairs)


# ═══════════════════════════════════════════════════════════
# STEP 4：合并字节词典
# ═══════════════════════════════════════════════════════════

def merge_byte_vocab(
    pair: Tuple[int, int],
    vocab: Dict[Tuple[int, ...], int],
) -> Dict[Tuple[int, ...], int]:
    """
    合并词典中所有词里出现的指定字节对。

    例子：
        pair = (108, 111)  ← 'l' + 'o' 的字节（108='l', 111='o'）
        输入词：(108, 111, 119)  ← 'l', 'o', 'w'
        输出词：(11119, 119)  ← 合并后 'lo' 作为单一 token id

    等等，BBPE 的 token 不再是字节，而是字节序列！
    合并后的新 token 是两个字节的序列，用一个新的元组来表示。

    实际实现里，合并后的 token 用一个新 id（> 255）表示，
    这里为了清晰用元组表示 token 内容。

    Args:
        pair: 要合并的字节对
        vocab: 当前词典

    Returns:
        更新后的词典
    """
    new_vocab: Dict[Tuple[int, ...], int] = {}
    a, b = pair

    for byte_seq, freq in vocab.items():
        # 在 byte_seq 中找所有相邻的 (a, b) 并合并
        new_seq: List[int] = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and byte_seq[i] == a and byte_seq[i + 1] == b:
                # 合并：用一个新 token id 表示这对字节
                # 这里用 (a << 8) | b 作为新 id（仅用于演示，实际用词表索引）
                merged_id = (a << 8) | b  # 临时 id，仅演示用
                new_seq.append(merged_id)
                i += 2
            else:
                new_seq.append(byte_seq[i])
                i += 1
        new_vocab[tuple(new_seq)] = freq

    return new_vocab


# ═══════════════════════════════════════════════════════════
# STEP 5：BBPE 训练
# ═══════════════════════════════════════════════════════════

def train_bbpe(
    corpus: List[str],
    target_vocab_size: int,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, int]]]:
    """
    BBPE 训练主函数。

    初始词表：256 个单字节 token（0-255），对应所有可能的字节值。
    然后在此基础上做 BPE 合并，直到词表大到目标大小。

    Args:
        corpus: 文本语料列表
        target_vocab_size: 目标词表大小（>= 256，因为初始就有 256 个字节 token）
        verbose: 是否打印训练过程

    Returns:
        token2id: 词表（字节序列元组 → id）
        merge_rules: 有序合并规则列表
    """
    assert target_vocab_size >= 256, "BBPE 词表最小为 256（初始字节 token 数）"

    # 初始词表：256 个单字节 token
    # 用字节值元组 (b,) 作为 key
    token2id: Dict[Tuple[int, ...], int] = {}
    for b in range(256):
        token2id[(b,)] = b

    # 把语料转为字节词典
    vocab = get_byte_vocab(corpus)

    merge_rules: List[Tuple[int, int]] = []
    step = 0

    while len(token2id) < target_vocab_size:
        # 统计字节 bigram 频率
        pairs = get_byte_stats(vocab)
        if not pairs:
            break

        # 选最高频 bigram
        best_pair = max(pairs, key=lambda p: (pairs[p], p))
        best_freq = pairs[best_pair]

        # 合并词典
        vocab = merge_byte_vocab(best_pair, vocab)

        # 新 token 是两个字节合并后的序列
        a, b = best_pair
        new_token_id = (a << 8) | b  # 仅演示用的临时 id
        # 实际应该是 256 + step
        actual_new_id = 256 + step
        token2id[(a, b)] = actual_new_id  # 用字节对作为新 token 的 key

        merge_rules.append(best_pair)

        step += 1
        if verbose:
            a_char = chr(a) if 32 <= a < 127 else f'<{a}>'
            b_char = chr(b) if 32 <= b < 127 else f'<{b}>'
            print(f"  Step {step:3d}: 合并字节 ({a},{b}) → '{a_char}{b_char}'  "
                  f"(频率={best_freq}, 词表={len(token2id)})")

    return token2id, merge_rules


# ═══════════════════════════════════════════════════════════
# STEP 6：BBPE 编码和解码
# ═══════════════════════════════════════════════════════════

def bbpe_encode(
    text: str,
    merge_rules: List[Tuple[int, int]],
    base_vocab_size: int = 256,
) -> List[int]:
    """
    用 BBPE 对文本编码。

    流程：
        1. 文本 → UTF-8 字节列表
        2. 按照 merge_rules 顺序合并字节对
        3. 返回 token id 列表

    Args:
        text: 输入文本（任意 Unicode，不会 OOV）
        merge_rules: 训练时得到的合并规则
        base_vocab_size: 基础字节 token 数（通常 256）

    Returns:
        token id 列表
    """
    # Step 1：文本 → 字节序列
    byte_ids: List[int] = text_to_bytes(text)

    # Step 2：按合并规则依次合并
    # 合并后的 token 用合成 id 表示：256 + 规则索引
    for rule_idx, (a, b) in enumerate(merge_rules):
        merged_id = base_vocab_size + rule_idx  # 这条规则生成的新 token id

        new_ids: List[int] = []
        i = 0
        while i < len(byte_ids):
            if i < len(byte_ids) - 1 and byte_ids[i] == a and byte_ids[i + 1] == b:
                new_ids.append(merged_id)
                i += 2
            else:
                new_ids.append(byte_ids[i])
                i += 1
        byte_ids = new_ids

    return byte_ids


def bbpe_decode(ids: List[int], merge_rules: List[Tuple[int, int]]) -> str:
    """
    BBPE 解码：token id 列表 → 原始文本。

    流程：
        1. 对每个 token id，还原为字节序列
        2. 把所有字节拼接
        3. UTF-8 解码为文本

    Args:
        ids: token id 列表
        merge_rules: 训练时的合并规则

    Returns:
        解码后的文本
    """
    # 构建 id → 字节序列 的映射
    # 基础字节：id 0-255 直接对应字节值
    id_to_bytes: Dict[int, List[int]] = {b: [b] for b in range(256)}

    # 合并 token：还原为对应的字节序列
    for rule_idx, (a, b) in enumerate(merge_rules):
        merged_id = 256 + rule_idx
        # 新 token 的字节序列 = token_a 的字节序列 + token_b 的字节序列
        id_to_bytes[merged_id] = id_to_bytes[a] + id_to_bytes[b]

    # 把所有 token 还原为字节
    all_bytes: List[int] = []
    for token_id in ids:
        all_bytes.extend(id_to_bytes.get(token_id, []))

    return bytes_to_text(all_bytes)


# ═══════════════════════════════════════════════════════════
# STEP 7：字节 → 可读 token 字符串
# ═══════════════════════════════════════════════════════════

def id_to_token_str(token_id: int, merge_rules: List[Tuple[int, int]]) -> str:
    """
    把 token id 转为可读字符串（用于调试）。

    对于字节 token（id 0-255），尝试用 UTF-8 解码；
    对于合并 token（id >= 256），还原字节序列后解码。

    Args:
        token_id: token id
        merge_rules: 训练时的合并规则

    Returns:
        可读字符串
    """
    # 构建 id → 字节序列 映射（同 decode）
    id_to_bytes: Dict[int, List[int]] = {b: [b] for b in range(256)}
    for rule_idx, (a, b) in enumerate(merge_rules):
        merged_id = 256 + rule_idx
        id_to_bytes[merged_id] = id_to_bytes.get(a, [a]) + id_to_bytes.get(b, [b])

    byte_seq = id_to_bytes.get(token_id, [])
    try:
        return bytes(byte_seq).decode('utf-8')
    except UnicodeDecodeError:
        # 字节序列不是合法 UTF-8（跨字符边界的 token）
        return f'bytes({byte_seq})'


# ═══════════════════════════════════════════════════════════
# 主程序演示
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("BBPE 字节级表示演示")
    print("=" * 60)

    # 演示不同字符的 UTF-8 字节表示
    examples = [
        ('A', '英文字母，1 字节（ASCII）'),
        ('你', '常见汉字，3 字节（BMP）'),
        ('𠀀', '生僻汉字，4 字节（超出 BMP）'),
        ('🌍', 'Emoji，4 字节（超出 BMP）'),
        ('á', '拉丁扩展字母，2 字节'),
    ]

    print("\n各字符的 UTF-8 字节表示：")
    print(f"{'字符':8} {'Unicode':12} {'字节(十六进制)':25} {'字节(十进制)'}")
    print("-" * 75)
    for char, desc in examples:
        utf8_bytes = text_to_bytes(char)
        hex_str = ' '.join(f'0x{b:02X}' for b in utf8_bytes)
        dec_str = str(utf8_bytes)
        cp = f'U+{ord(char):04X}'
        print(f"{char:8} {cp:12} {hex_str:25} {dec_str}  ← {desc}")

    print("\n" + "=" * 60)
    print("关键理解：0xF0 = 240，它们是同一个字节的两种写法")
    print("=" * 60)
    print("\n  十六进制 → 十进制 对照：")
    demo_pairs = [(0xF0, '生僻字第1字节'), (0xA0, '生僻字第2字节'),
                  (0x80, '生僻字第3/4字节'), (0xE4, '\"你\"第1字节'),
                  (0xBD, '\"你\"第2字节'), (0xA0, '\"你\"第3字节')]
    for hex_val, desc in demo_pairs:
        print(f"  0x{hex_val:02X} = {hex_val:3d}  ← {desc}")

    print("\n" + "=" * 60)
    print("BBPE 训练演示（中英文混合语料）")
    print("=" * 60)

    corpus = [
        "hello world hello",
        "你好世界 你好",
        "hello 你好 world",
        "machine learning 机器学习",
        "natural language processing 自然语言处理",
    ]

    print(f"\n训练语料（{len(corpus)} 条）：")
    for line in corpus:
        print(f"  {line}")

    print("\n开始训练（目标词表 = 280，即 256 个字节 + 24 个合并 token）：")
    token2id, merge_rules = train_bbpe(corpus, target_vocab_size=280, verbose=True)

    print(f"\n词表大小：{len(token2id)}（包含 {len(merge_rules)} 条合并规则）")

    print("\n" + "=" * 60)
    print("BBPE 编码测试")
    print("=" * 60)

    test_texts = [
        "hello",      # 纯英文
        "你好",        # 中文
        "𠀀",          # 生僻字（传统 BPE 会 OOV，BBPE 不会！）
        "hello 你好",  # 混合
    ]

    print()
    for text in test_texts:
        ids = bbpe_encode(text, merge_rules)
        token_strs = [id_to_token_str(i, merge_rules) for i in ids]
        decoded = bbpe_decode(ids, merge_rules)
        round_trip_ok = decoded == text
        print(f"  输入  : '{text}'  (UTF-8 字节: {text_to_bytes(text)})")
        print(f"  tokens: {token_strs}")
        print(f"  ids   : {ids}")
        print(f"  解码  : '{decoded}'  往返: {'✓' if round_trip_ok else '✗'}")
        print()

    print("关键点：即使是 '𠀀' 这样的生僻字，BBPE 也能正确编解码（无 OOV）！")

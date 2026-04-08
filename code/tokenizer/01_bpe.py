"""
BPE（Byte Pair Encoding）完整实现
======================================
从零实现 GPT 系列使用的分词算法，包含：
  1. 语料 → 字符级词频词典
  2. 统计 bigram 频率
  3. 迭代合并，构建词表 + 合并规则
  4. 推理：用合并规则对新文本编码
  5. 解码：token id 序列 → 原始文本

核心论文：Sennrich et al. 2016, "Neural Machine Translation of Rare Words with Subword Units"
"""

from collections import Counter
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════
# STEP 1：把语料变成字符级词频词典
# ═══════════════════════════════════════════════════════════

def get_vocab(corpus: List[str]) -> Dict[str, int]:
    """
    将文本语料转换为「字符级词频词典」。

    BPE 的分词单位是「词」，每个词被拆成字符序列。
    约定：在每个词末尾加 </w> 标记，表示"这是词的结尾"。
    解码时看到 </w> 就知道后面应该跟空格（或结束）。

    例子：
        输入语料：["low low newest"]
        输出词典：{
            'l o w </w>': 2,     ← "low" 出现 2 次，拆成字符
            'n e w e s t </w>': 1
        }

    Args:
        corpus: 文本列表，每个元素是一段文本

    Returns:
        词频词典：{字符序列字符串: 出现次数}
    """
    vocab: Dict[str, int] = Counter()
    for text in corpus:
        for word in text.strip().split():
            # list(word) 把词拆成单字符列表，末尾加 </w>
            # ' '.join(...) 让字符间有空格，方便后续按空格 split 出 symbol 列表
            char_seq = ' '.join(list(word)) + ' </w>'
            vocab[char_seq] += 1
    return dict(vocab)


# ═══════════════════════════════════════════════════════════
# STEP 2：统计所有相邻 symbol 对（bigram）的频率
# ═══════════════════════════════════════════════════════════

def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    遍历词典中所有词，统计相邻 symbol 对的总频率。

    例子：
        词典：{'l o w </w>': 5, 'n e w e s t </w>': 6}
        输出：{
            ('l', 'o'): 5,
            ('o', 'w'): 5,
            ('w', '</w>'): 5,
            ('n', 'e'): 6,
            ('e', 'w'): 6,
            ('w', 'e'): 6,
            ('e', 's'): 6,
            ('s', 't'): 6,
            ('t', '</w>'): 6,
        }

    Args:
        vocab: 字符级词频词典

    Returns:
        bigram 频率词典：{(symbol_a, symbol_b): 出现总次数}
    """
    pairs: Dict[Tuple[str, str], int] = Counter()
    for word, freq in vocab.items():
        symbols = word.split()  # 按空格分割出 symbol 列表
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


# ═══════════════════════════════════════════════════════════
# STEP 3：把词典中所有词里的指定 pair 合并为新 token
# ═══════════════════════════════════════════════════════════

def merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    """
    将词典中所有出现 pair 的位置，把两个相邻 symbol 合并成一个新 token。

    例子：
        pair = ('e', 's')
        词典前：{'n e w e s t </w>': 6}
        词典后：{'n e w es t </w>': 6}   ← 'e' 和 's' 合并成 'es'

    实现方式：字符串替换（用空格区分 token 边界）
        把 " e s " 替换为 " es "，避免误替换 "es" 中间的字母

    Args:
        pair: 要合并的 bigram，如 ('e', 's')
        vocab: 当前词典

    Returns:
        更新后的词典
    """
    # 目标字符串：" e s "（两侧加空格保证精确匹配）
    bigram_str = ' '.join(pair)           # "e s"
    merged_token = ''.join(pair)          # "es"

    new_vocab: Dict[str, int] = {}
    for word, freq in vocab.items():
        # replace 会替换字符串中所有匹配的 bigram
        new_word = word.replace(bigram_str, merged_token)
        new_vocab[new_word] = freq
    return new_vocab


# ═══════════════════════════════════════════════════════════
# STEP 4：训练主函数
# ═══════════════════════════════════════════════════════════

def train_bpe(
    corpus: List[str],
    target_vocab_size: int,
    verbose: bool = False,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    BPE 训练主函数：从语料出发，迭代合并，直到词表达到目标大小。

    算法流程：
        初始词表 = 语料中出现过的所有字符 + '</w>'
        循环：
            1. 统计所有 bigram 频率
            2. 选出频率最高的 bigram（贪心）
            3. 合并词典中所有该 bigram → 新 token 加入词表
            4. 记录合并规则
        直到词表大小 >= target_vocab_size

    Args:
        corpus: 文本语料列表
        target_vocab_size: 目标词表大小（包括初始字符 token）
        verbose: 是否打印每步合并过程

    Returns:
        token2id: 词表，{token字符串: id整数}
        merge_rules: 有序合并规则列表（推理时按此顺序应用）
    """
    # 初始化字符级词典
    vocab = get_vocab(corpus)

    # 构建初始词表：所有出现过的单字符 + '</w>'
    # sorted 保证 id 分配的确定性（不随机）
    all_symbols: set = set()
    for word in vocab:
        all_symbols.update(word.split())
    token2id: Dict[str, int] = {sym: i for i, sym in enumerate(sorted(all_symbols))}

    merge_rules: List[Tuple[str, str]] = []

    # 迭代合并，直到词表达到目标大小
    step = 0
    while len(token2id) < target_vocab_size:
        pairs = get_stats(vocab)
        if not pairs:
            # 所有词都已经是单个 token，无法继续合并
            break

        # 选频率最高的 pair（贪心）
        # 如果有多个 pair 频率相同，max 会按字典序取第一个（保证确定性）
        best_pair = max(pairs, key=lambda p: (pairs[p], p))
        best_freq = pairs[best_pair]

        # 合并词典
        vocab = merge_vocab(best_pair, vocab)

        # 新 token = 两个 symbol 拼接
        new_token = ''.join(best_pair)
        merge_rules.append(best_pair)
        token2id[new_token] = len(token2id)

        step += 1
        if verbose:
            print(f"  Step {step:3d}: 合并 {best_pair} → '{new_token}'  (频率={best_freq}, 词表大小={len(token2id)})")

    return token2id, merge_rules


# ═══════════════════════════════════════════════════════════
# STEP 5：推理 - 对单个词应用合并规则
# ═══════════════════════════════════════════════════════════

def bpe_tokenize_word(word: str, merge_rules: List[Tuple[str, str]]) -> List[str]:
    """
    对单个词（不含空格）按顺序应用训练时学到的合并规则，返回 token 字符串列表。

    关键：合并规则的顺序很重要！
    高频的 bigram 先合并，所以 merge_rules[0] 是最高频的合并。
    推理时必须按相同顺序应用，才能复现训练时的分词结果。

    例子：
        word = "low"，merge_rules 包含 [('l','o'), ('lo','w'), ('low','</w>'), ...]
        初始：['l', 'o', 'w', '</w>']
        应用 ('l','o') → ['lo', 'w', '</w>']
        应用 ('lo','w') → ['low', 'w', '</w>']  ← 注意：第二个 'w' 不参与这条规则
        应用 ('low','</w>') → ['low</w>']
        最终：['low</w>']

    Args:
        word: 单个词（不含空格）
        merge_rules: 训练时得到的有序合并规则列表

    Returns:
        token 字符串列表
    """
    # 初始化：每个字符是一个 token，末尾加 '</w>'
    tokens: List[str] = list(word) + ['</w>']

    for pair in merge_rules:
        # 扫描当前 token 列表，找到所有匹配的相邻 pair 并合并
        i = 0
        new_tokens: List[str] = []
        while i < len(tokens):
            # 当前位置和下一位置恰好等于目标 pair → 合并
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2  # 跳过已合并的两个元素
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        # 优化：如果只剩一个 token，后续规则都不会再合并了
        if len(tokens) == 1:
            break

    return tokens


# ═══════════════════════════════════════════════════════════
# STEP 6：编码 - 文本 → token id 列表
# ═══════════════════════════════════════════════════════════

def bpe_encode(
    text: str,
    token2id: Dict[str, int],
    merge_rules: List[Tuple[str, str]],
    unk_token: str = '<unk>',
) -> List[int]:
    """
    将任意文本编码为 token id 列表。

    流程：
        1. 按空格切分文本为词列表
        2. 对每个词调用 bpe_tokenize_word 得到 token 字符串列表
        3. 查词表得到 id；未知 token 用 unk_id 替代

    注意：
        - 这里假设输入文本已经预处理（如小写化）
        - 真实的 GPT tokenizer 还会在词前加空格前缀（如 'Ġhello'）来区分词首和词中
        - 本实现为了清晰省略了这个细节

    Args:
        text: 待编码文本
        token2id: 词表
        merge_rules: 训练时的合并规则
        unk_token: 未知 token 的字符串表示

    Returns:
        token id 列表
    """
    unk_id = token2id.get(unk_token, 0)
    ids: List[int] = []

    for word in text.strip().split():
        word_tokens = bpe_tokenize_word(word, merge_rules)
        for t in word_tokens:
            ids.append(token2id.get(t, unk_id))

    return ids


# ═══════════════════════════════════════════════════════════
# STEP 7：解码 - token id 列表 → 原始文本
# ═══════════════════════════════════════════════════════════

def bpe_decode(
    ids: List[int],
    id2token: Dict[int, str],
    unk_token: str = '<unk>',
) -> str:
    """
    将 token id 列表解码回原始文本。

    解码规则：
        - 把所有 token 字符串拼接
        - </w> 标记词尾，替换为空格
        - 去掉首尾多余空格

    例子：
        ids = [id('low</w>'), id('n'), id('e'), id('w'), id('est</w>')]
        拼接：'low</w>newest</w>'
        替换：'low newest '
        strip：'low newest'

    Args:
        ids: token id 列表
        id2token: id → token 字符串的反向词表
        unk_token: 未知 id 用这个替代

    Returns:
        解码后的文本字符串
    """
    tokens = [id2token.get(i, unk_token) for i in ids]
    # 拼接所有 token，把 </w> 替换为空格，最后 strip
    text = ''.join(tokens).replace('</w>', ' ').strip()
    return text


# ═══════════════════════════════════════════════════════════
# 主程序：完整演示
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 经典 BPE 教学语料（来自原论文示例）
    corpus = [
        "low low low low low",           # "low" 出现 5 次
        "lower lower",                   # "lower" 出现 2 次
        "newest newest newest newest newest newest",  # "newest" 出现 6 次
        "widest widest widest",          # "widest" 出现 3 次
    ]

    print("=" * 60)
    print("BPE 训练过程")
    print("=" * 60)

    # 先看看初始词典长什么样
    initial_vocab = get_vocab(corpus)
    print("\n初始字符级词典：")
    for word, freq in sorted(initial_vocab.items(), key=lambda x: -x[1]):
        print(f"  '{word}' : {freq}")

    print("\n开始训练（目标词表大小 = 30）：")
    token2id, merge_rules = train_bpe(corpus, target_vocab_size=30, verbose=True)
    id2token = {v: k for k, v in token2id.items()}

    print(f"\n最终词表（{len(token2id)} 个 token）：")
    # 按 id 顺序打印
    for token, idx in sorted(token2id.items(), key=lambda x: x[1]):
        print(f"  {idx:3d}: '{token}'")

    print("\n" + "=" * 60)
    print("BPE 编码测试")
    print("=" * 60)

    test_cases = [
        "low newest",          # 训练集中出现过
        "lower widest",        # 训练集中出现过
        "lowest",              # 训练集中未出现，但子词 'low' 'est' 都有
    ]

    for text in test_cases:
        ids = bpe_encode(text, token2id, merge_rules)
        tokens = [id2token.get(i, '<unk>') for i in ids]
        decoded = bpe_decode(ids, id2token)
        print(f"\n  输入  : '{text}'")
        print(f"  tokens: {tokens}")
        print(f"  ids   : {ids}")
        print(f"  解码  : '{decoded}'")
        print(f"  往返一致: {'✓' if decoded == text else '✗'}")

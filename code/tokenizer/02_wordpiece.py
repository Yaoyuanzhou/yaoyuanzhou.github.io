"""
WordPiece 完整实现（BERT 风格）
======================================
BERT 使用的分词算法，与 BPE 的主要区别：
  - BPE：按 bigram 频率合并
  - WordPiece：按 PMI（点互信息）打分，选择合并后最大化语言模型概率的 pair

推理时使用「最长匹配优先（MaxMatch / Greedy Longest First）」算法。
非首子词用 ## 前缀标记，表示紧跟前一个 token（无空格）。

核心论文：
  - Schuster & Nakamura 2012（原版 WordPiece）
  - Devlin et al. 2018 BERT（广泛使用）
"""

from collections import Counter
from typing import Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════
# PART 1：WordPiece 训练（PMI 打分版）
# ═══════════════════════════════════════════════════════════

def train_wordpiece(
    corpus: List[str],
    target_vocab_size: int,
    special_tokens: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[str]:
    """
    从语料训练 WordPiece 词表。

    与 BPE 的关键区别：
        BPE 选的是"出现次数最多"的 pair
        WordPiece 选的是"PMI 得分最高"的 pair

    PMI（点互信息）公式：
        score(A, B) = P(AB) / (P(A) × P(B))

    直觉：如果 A 和 B 总是一起出现（不是独立随机凑的），就应该合并。
    例子：
        'un' + 'happy' = 50 次，但 'un' 出现 1000 次，'happy' 出现 800 次
        → 它们凑巧共现，PMI 较低，不急于合并
        'un' + 'able' = 300 次，'un' 1000 次，'able' 350 次
        → 几乎每次 'un' 都跟着 'able'，PMI 高，应该合并

    ## 前缀约定：
        词的第一个字符：直接用（如 'u'）
        词的非首字符：加 ## 前缀（如 '##n'，'##able'）
        解码时看到 ## 就直接拼接，不加空格

    Args:
        corpus: 文本语料列表
        target_vocab_size: 目标词表大小
        special_tokens: 特殊 token 列表（[PAD], [UNK] 等）
        verbose: 是否打印训练过程

    Returns:
        词表列表（顺序即 id）
    """
    if special_tokens is None:
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    # ── Step 1：统计词频，构建字符级 symbol 序列 ──────────────
    word_counts: Dict[str, int] = Counter()
    char_counts: Dict[str, int] = Counter()

    for text in corpus:
        for word in text.lower().split():
            word_counts[word] += 1
            for i, ch in enumerate(word):
                # 首字符不加前缀，非首字符加 ## 前缀
                symbol = ch if i == 0 else '##' + ch
                char_counts[symbol] += 1

    # 初始词表 = 特殊 token + 所有字符 symbol
    vocab: Set[str] = set(special_tokens) | set(char_counts.keys())

    # 每个词对应的当前 symbol 序列（随合并迭代更新）
    # 初始：每个词被拆成字符级 symbol 列表
    word_symbols: Dict[str, List[str]] = {}
    for word in word_counts:
        symbols = [word[0]]  # 首字符不加前缀
        for ch in word[1:]:
            symbols.append('##' + ch)  # 其余字符加 ## 前缀
        word_symbols[word] = symbols

    # ── Step 2：迭代 PMI 合并 ──────────────────────────────────
    step = 0
    while len(vocab) < target_vocab_size:

        # 统计当前 symbol 频率 和 symbol pair 频率
        sym_freq: Dict[str, int] = Counter()
        pair_freq: Dict[tuple, int] = Counter()

        for word, count in word_counts.items():
            syms = word_symbols[word]
            for s in syms:
                sym_freq[s] += count
            for i in range(len(syms) - 1):
                pair_freq[(syms[i], syms[i + 1])] += count

        if not pair_freq:
            break  # 所有词都是单 token，无法继续合并

        total = sum(sym_freq.values())

        # PMI 打分：score = P(AB) / (P(A) * P(B))
        # = [count(AB)/total] / ([count(A)/total] * [count(B)/total])
        # = count(AB) * total / (count(A) * count(B))
        best_pair, best_score = None, -1.0
        for (a, b), ab_count in pair_freq.items():
            if sym_freq[a] == 0 or sym_freq[b] == 0:
                continue
            score = (ab_count * total) / (sym_freq[a] * sym_freq[b])
            if score > best_score:
                best_score, best_pair = score, (a, b)

        if best_pair is None:
            break

        # 生成新 token
        # 规则：如果 b 以 ## 开头，合并后去掉 b 的 ## 前缀
        # 例子：'un' + '##ab' → 'unab'（不是 'un##ab'）
        a, b = best_pair
        b_suffix = b[2:] if b.startswith('##') else b  # 去掉 b 的 ## 前缀
        new_token = a + b_suffix
        vocab.add(new_token)

        # 更新所有词的 symbol 序列（把 a+b 合并为 new_token）
        for word in word_symbols:
            syms = word_symbols[word]
            merged: List[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            word_symbols[word] = merged

        step += 1
        if verbose:
            print(f"  Step {step:3d}: {a!r} + {b!r} → {new_token!r}  (PMI={best_score:.2f}, 词表={len(vocab)})")

    # 把词表从 set 转成稳定排序的列表（特殊 token 在前）
    vocab_list = special_tokens[:]
    for t in sorted(vocab - set(special_tokens)):
        vocab_list.append(t)
    return vocab_list


# ═══════════════════════════════════════════════════════════
# PART 2：WordPiece 推理 - 最长匹配算法（MaxMatch）
# ═══════════════════════════════════════════════════════════

class WordPieceTokenizer:
    """
    BERT 风格的 WordPiece 分词器。

    核心推理算法：最长匹配优先（Greedy Longest Match）
        给定词表和一个词，从词的开头开始，
        每次尽量匹配最长的词表中的 token。

    例子：词 "unaffordable"，词表中有 'un', '##aff', '##ord', '##able'
        从位置 0 开始，往右找最长匹配：
            "unaffordable" → 不在词表
            "unaffordabl"  → 不在词表
            ...
            "un"           → 在词表！取 'un'，移动到位置 2
        从位置 2 开始（非词首，加 ## 前缀匹配）：
            "##affordable" → 不在词表
            ...
            "##aff"        → 在词表！取 '##aff'，移动到位置 5
        ...直到处理完整个词
        最终：['un', '##aff', '##ord', '##able']
    """

    def __init__(
        self,
        vocab: List[str],
        unk_token: str = '[UNK]',
        max_chars_per_word: int = 200,
    ):
        """
        Args:
            vocab: 词表列表，顺序即 id
            unk_token: 未知 token（当词无法被词表中任何 token 覆盖时使用）
            max_chars_per_word: 超过此长度的词直接标记为 [UNK]（防止超长字符串 O(n²) 爆炸）
        """
        self.vocab: Set[str] = set(vocab)
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(vocab)}
        self.unk_token = unk_token
        self.unk_id = self.token2id.get(unk_token, 0)
        self.max_chars_per_word = max_chars_per_word

    def _tokenize_word(self, word: str) -> List[str]:
        """
        对单个词（不含空格）做最长匹配分词。

        当某个位置找不到任何匹配时，整个词标记为 [UNK]。
        这和 BPE 不同：BPE 可以退化到字符级；WordPiece 直接 [UNK]。

        时间复杂度：O(n²)，n 为词的字符数。

        Args:
            word: 单个词

        Returns:
            token 字符串列表。如果整个词无法匹配，返回 [self.unk_token]
        """
        # 超长词直接返回 [UNK]，避免 O(n²) 超时
        if len(word) > self.max_chars_per_word:
            return [self.unk_token]

        tokens: List[str] = []
        start = 0
        word_len = len(word)

        while start < word_len:
            end = word_len  # 从最长子串开始往短处找
            found_token = None

            while start < end:
                substr = word[start:end]
                # 非词首子串要加 ## 前缀再查词表
                candidate = substr if start == 0 else '##' + substr

                if candidate in self.vocab:
                    found_token = candidate
                    break  # 找到最长匹配，停止缩短
                end -= 1

            if found_token is None:
                # 当前位置起没有任何子串能在词表中找到
                # 整个词标记为 [UNK]
                return [self.unk_token]

            tokens.append(found_token)
            start = end  # 移动到已匹配部分之后

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        文本 → token 字符串列表（不加特殊 token）。

        用于调试，可以看到具体切分结果。

        Args:
            text: 输入文本

        Returns:
            token 字符串列表
        """
        result: List[str] = []
        for word in text.lower().split():
            result.extend(self._tokenize_word(word))
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        """
        文本 → token id 列表。

        Args:
            text: 输入文本
            add_special_tokens: 是否在首尾加 [CLS] 和 [SEP]
            max_length: 最大长度（超过截断，不足可 padding）
            padding: 是否用 [PAD] 补到 max_length

        Returns:
            token id 列表
        """
        tokens: List[str] = []

        if add_special_tokens:
            tokens.append('[CLS]')

        for word in text.lower().split():
            tokens.extend(self._tokenize_word(word))

        if add_special_tokens:
            tokens.append('[SEP]')

        ids = [self.token2id.get(t, self.unk_id) for t in tokens]

        # 截断
        if max_length is not None:
            ids = ids[:max_length]
            # padding
            if padding and len(ids) < max_length:
                pad_id = self.token2id.get('[PAD]', 0)
                ids += [pad_id] * (max_length - len(ids))

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        token id 列表 → 原始文本。

        解码关键：遇到 ## 前缀的 token 直接拼接，不加空格。

        例子：
            tokens = ['hello', '##world', '[SEP]']
            skip_special_tokens=True → 'helloworld'

        Args:
            ids: token id 列表
            skip_special_tokens: 是否跳过 [CLS], [SEP], [PAD], [MASK] 等

        Returns:
            解码后的文本
        """
        SPECIAL = {'[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]'}
        words: List[str] = []
        current_word = ''

        for token_id in ids:
            token = self.id2token.get(token_id, self.unk_token)

            # 跳过特殊 token
            if skip_special_tokens and token in SPECIAL:
                if current_word:
                    words.append(current_word)
                    current_word = ''
                continue

            if token.startswith('##'):
                # ## 前缀：直接拼接到当前词，不加空格
                current_word += token[2:]
            else:
                # 新词开始：保存之前的词，开始新词
                if current_word:
                    words.append(current_word)
                current_word = token

        if current_word:
            words.append(current_word)

        return ' '.join(words)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        批量编码（返回 input_ids + attention_mask）。

        attention_mask：
            1 = 真实 token（模型应该关注）
            0 = PAD token（模型应该忽略）

        Args:
            texts: 多条文本列表
            max_length: 最大长度（所有序列统一截断/补齐到此长度）
            padding: 是否补 PAD
            add_special_tokens: 是否加 [CLS]/[SEP]

        Returns:
            {'input_ids': [...], 'attention_mask': [...]}
        """
        pad_id = self.token2id.get('[PAD]', 0)
        all_ids: List[List[int]] = []
        all_masks: List[List[int]] = []

        for text in texts:
            ids = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
            )
            # attention_mask：非 PAD 位置为 1
            mask = [0 if i == pad_id else 1 for i in ids]
            all_ids.append(ids)
            all_masks.append(mask)

        return {'input_ids': all_ids, 'attention_mask': all_masks}


# ══════════════════════════════════════════════════════════
# 主程序演示
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("WordPiece 训练演示")
    print("=" * 60)

    corpus = [
        "unaffordable hello world",
        "unhappy happiness happy",
        "tokenization token tokens",
        "learning learned learner",
        "running runner runs",
    ]
    print("\n训练语料：")
    for line in corpus:
        print(f"  {line}")

    print("\n开始训练（目标词表 = 60）：")
    vocab_list = train_wordpiece(corpus, target_vocab_size=60, verbose=True)

    print(f"\n最终词表（{len(vocab_list)} 个 token）：")
    for i, token in enumerate(vocab_list):
        print(f"  {i:3d}: '{token}'")

    print("\n" + "=" * 60)
    print("WordPiece 推理演示")
    print("=" * 60)

    # 用一个预定义的小词表演示（避免训练词表变动影响演示）
    demo_vocab = [
        '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
        # 字符
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i',
        '##j', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
        '##t', '##u', '##v', '##w', '##x', '##y', '##z',
        # 合并后的 token（模拟训练结果）
        'un', '##aff', '##ord', '##able', 'hello', 'world',
        '##ing', '##ed', '##er', '##ness', '##tion',
        'token', 'happy', 'learn',
    ]

    tokenizer = WordPieceTokenizer(demo_vocab)

    # 测试分词
    test_words = ['unaffordable', 'hello', 'tokenization', 'xyz999']
    print("\n单词分词测试：")
    for word in test_words:
        tokens = tokenizer._tokenize_word(word)
        print(f"  '{word}' → {tokens}")

    # 测试完整编码解码
    print("\n完整 encode / decode 测试：")
    texts = [
        "hello world",
        "unaffordable token",
    ]
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        tokens = [tokenizer.id2token.get(i, '[UNK]') for i in ids]
        decoded = tokenizer.decode(ids)
        print(f"\n  输入  : '{text}'")
        print(f"  tokens: {tokens}")
        print(f"  ids   : {ids}")
        print(f"  解码  : '{decoded}'")

    # 批量编码测试
    print("\n批量 encode 测试（max_length=10, padding=True）：")
    batch = tokenizer.batch_encode(
        ["hello world", "unaffordable"],
        max_length=10,
        padding=True,
    )
    for i, (ids, mask) in enumerate(zip(batch['input_ids'], batch['attention_mask'])):
        print(f"  [{i}] ids : {ids}")
        print(f"  [{i}] mask: {mask}")

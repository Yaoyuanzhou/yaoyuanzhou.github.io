"""
Unigram Language Model 分词器完整实现
======================================
T5、XLNet、mBART 使用的分词算法。

与 BPE/WordPiece 的根本区别：
  - BPE/WordPiece：从小词表出发，不断合并（增量式）
  - Unigram LM：从大词表出发，不断剪枝（减量式）

核心思想：
  - 假设分词是一个概率模型：P(分词方案) = ∏ P(每个 token)
  - 词表中每个 token 有一个概率 P(token)
  - 对于一段文本，用 Viterbi 算法找最优分词（概率最大的方案）
  - 剪枝：移除那些"删掉后整体语料的概率损失最小"的 token

核心论文：
  - Kudo 2018, "Subword Regularization: Improving Neural Network Translation Models
    with Multiple Subword Candidates"
  - Kudo & Richardson 2018, "SentencePiece: A simple and language independent
    subword tokenizer and detokenizer for Neural Text Processing"
"""

import math
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════
# STEP 1：构建初始候选词表（从语料中提取所有子串）
# ═══════════════════════════════════════════════════════════

def build_initial_vocab(
    corpus: List[str],
    max_vocab_size: int = 5000,
    min_freq: int = 2,
) -> Dict[str, float]:
    """
    从语料中提取所有子串作为候选 token，按词频估算初始概率。

    真实实现（如 SentencePiece）会枚举更多候选，这里简化为：
      - 提取所有 1-8 字符的子串
      - 过滤低频子串
      - 用相对词频作为初始概率

    Args:
        corpus: 文本语料列表
        max_vocab_size: 初始候选词表的最大大小
        min_freq: 子串最低出现频率（低于此过滤）

    Returns:
        候选词表：{token: log_prob}（使用对数概率，避免浮点下溢）
    """
    from collections import Counter

    # 统计所有子串出现次数
    substr_count: Dict[str, int] = Counter()
    for text in corpus:
        text = text.strip()
        for i in range(len(text)):
            for j in range(i + 1, min(i + 9, len(text) + 1)):  # 最长 8 字符
                substr = text[i:j]
                if substr.strip():  # 跳过纯空格子串
                    substr_count[substr] += 1

    # 过滤低频，保留 top-max_vocab_size
    filtered = {
        s: c for s, c in substr_count.items()
        if c >= min_freq
    }
    # 按频率排序，取 top N
    top_substrs = sorted(filtered.items(), key=lambda x: -x[1])[:max_vocab_size]

    total = sum(c for _, c in top_substrs)
    # 用对数概率：log(count / total)
    vocab = {s: math.log(c / total) for s, c in top_substrs}
    return vocab


# ═══════════════════════════════════════════════════════════
# STEP 2：Viterbi 算法 - 给定词表，找一个词的最优分词
# ═══════════════════════════════════════════════════════════

def viterbi_segment(
    text: str,
    vocab: Dict[str, float],
    unk_score: float = -20.0,
) -> Tuple[List[str], float]:
    """
    用 Viterbi（动态规划）找概率最大的分词方案。

    直觉：
        把文本想象成从位置 0 走到位置 n 的路径。
        每一步可以从位置 i 跳到位置 j（如果 text[i:j] 在词表里）。
        每一跳的"奖励"是 log P(text[i:j])（越大越好，即词越常见）。
        Viterbi 找奖励总和最大的路径。

    时间复杂度：O(n² × max_token_len)，n 为文本长度。

    Args:
        text: 待分词文本（单词，不含空格）
        vocab: {token: log_prob}
        unk_score: 未知 token 的惩罚分数（负数，越大越惩罚 OOV）

    Returns:
        (最优 token 列表, 总 log 概率)
    """
    n = len(text)
    # dp[i]：从位置 0 到位置 i 的最大对数概率
    # bp[i]：达到位置 i 的最优路径中，上一个切点的位置
    dp: List[float] = [-math.inf] * (n + 1)
    bp: List[int] = [-1] * (n + 1)
    dp[0] = 0.0  # 起点概率为 1，log(1) = 0

    # 考虑的最大 token 长度（避免扫描过长子串）
    max_len = max((len(t) for t in vocab), default=8)
    max_len = min(max_len, 16)  # 不超过 16，防止 O(n²) 过慢

    for i in range(1, n + 1):
        # 枚举所有可能的上一个切点 j
        for j in range(max(0, i - max_len), i):
            substr = text[j:i]
            score = vocab.get(substr, None)

            if score is None:
                # token 不在词表：
                # 如果是单字符，用 unk_score（必须接受，不然路径断了）
                # 如果是多字符子串且不在词表，跳过（等更短的子串来覆盖）
                if len(substr) == 1:
                    score = unk_score
                else:
                    continue  # 多字符 OOV 跳过

            candidate = dp[j] + score
            if candidate > dp[i]:
                dp[i] = candidate
                bp[i] = j  # 记录最优切点

    # 回溯 bp 数组，重建最优 token 序列
    tokens: List[str] = []
    pos = n
    while pos > 0:
        prev = bp[pos]
        if prev == -1:
            # 理论上不应该走到这里（单字符兜底保证了路径连通）
            break
        tokens.append(text[prev:pos])
        pos = prev
    tokens.reverse()  # 回溯得到的是逆序，需要翻转

    return tokens, dp[n]


# ═══════════════════════════════════════════════════════════
# STEP 3：EM 训练 - 估算每个 token 的真实概率
# ═══════════════════════════════════════════════════════════

def compute_token_expected_counts(
    corpus_words: List[str],
    vocab: Dict[str, float],
) -> Dict[str, float]:
    """
    给定当前词表概率，用 Viterbi 解码所有词，
    统计每个 token 的"期望出现次数"（EM 的 E 步简化版）。

    这里用 Viterbi（最优分词）代替完整的前后向算法（所有分词加权期望）。
    虽然是近似，但实践中效果相近且速度更快。

    Args:
        corpus_words: 语料中的所有词列表
        vocab: 当前词表 {token: log_prob}

    Returns:
        期望计数 {token: expected_count}
    """
    expected_counts: Dict[str, float] = {}
    for word in corpus_words:
        tokens, _ = viterbi_segment(word, vocab)
        for t in tokens:
            expected_counts[t] = expected_counts.get(t, 0.0) + 1.0
    return expected_counts


def update_vocab_probs(expected_counts: Dict[str, float]) -> Dict[str, float]:
    """
    根据期望计数重新估算对数概率（EM 的 M 步）。

    P(token) = count(token) / sum(count(all tokens))

    Args:
        expected_counts: {token: expected_count}

    Returns:
        更新后的 {token: log_prob}
    """
    total = sum(expected_counts.values())
    if total == 0:
        return {}
    return {t: math.log(c / total) for t, c in expected_counts.items()}


# ═══════════════════════════════════════════════════════════
# STEP 4：剪枝 - 移除"损失最小"的 token
# ═══════════════════════════════════════════════════════════

def compute_log_likelihood(
    corpus_words: List[str],
    vocab: Dict[str, float],
) -> float:
    """
    计算语料在当前词表下的总对数似然。

    L = sum over all words: log P(word) = sum over all words: best_seg_score

    Args:
        corpus_words: 语料词列表
        vocab: {token: log_prob}

    Returns:
        总对数似然（越大越好）
    """
    total_ll = 0.0
    for word in corpus_words:
        _, score = viterbi_segment(word, vocab)
        total_ll += score
    return total_ll


def prune_vocab(
    vocab: Dict[str, float],
    corpus_words: List[str],
    target_size: int,
    prune_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    每轮剪掉词表中"删除后语料损失最小"的 token，直到词表大小达标。

    剪枝策略：
        对每个 token t，计算"删掉 t 后语料对数似然的变化"：
            Δll(t) = ll(vocab) - ll(vocab without t)
        Δll 越小，说明 t 越"不重要"，越应该删掉。

    注意：单字符 token 不能删，否则路径可能断掉（OOV 无法兜底）。

    Args:
        vocab: 当前词表
        corpus_words: 语料词列表
        target_size: 目标词表大小
        prune_ratio: 每次剪掉的比例（太大则跳过好 token，太小则太慢）

    Returns:
        剪枝后的词表
    """
    current_ll = compute_log_likelihood(corpus_words, vocab)

    while len(vocab) > target_size:
        # 计算每个 token 的重要性得分（删除后的 ll 损失）
        scores: List[Tuple[float, str]] = []

        for token in vocab:
            # 单字符不可删（Viterbi 需要字符级兜底）
            if len(token) == 1:
                continue

            # 临时删掉这个 token，计算 ll 变化
            temp_vocab = {t: p for t, p in vocab.items() if t != token}
            new_ll = compute_log_likelihood(corpus_words, temp_vocab)
            delta = current_ll - new_ll  # 损失越小越可以删
            scores.append((delta, token))

        if not scores:
            break  # 没有可删的 token 了

        # 删掉损失最小的 prune_ratio 比例的 token
        scores.sort()  # 按 delta 升序（delta 越小越应该删）
        n_prune = max(1, int(len(scores) * prune_ratio))
        tokens_to_remove = {t for _, t in scores[:n_prune]}

        vocab = {t: p for t, p in vocab.items() if t not in tokens_to_remove}

        # 重新估算概率
        expected_counts = compute_token_expected_counts(corpus_words, vocab)
        vocab = update_vocab_probs(expected_counts)

        print(f"  剪枝后词表大小: {len(vocab)}")

    return vocab


# ═══════════════════════════════════════════════════════════
# STEP 5：Unigram LM 训练主函数
# ═══════════════════════════════════════════════════════════

def train_unigram_lm(
    corpus: List[str],
    target_vocab_size: int,
    initial_vocab_size: int = 2000,
    em_steps: int = 3,
) -> Dict[str, float]:
    """
    Unigram LM 训练主函数。

    流程：
        1. 从大词表出发（所有子串）
        2. EM 估算各 token 概率
        3. 剪枝掉不重要的 token
        4. 重复 2-3 直到词表大小达标

    Args:
        corpus: 文本语料
        target_vocab_size: 最终词表大小
        initial_vocab_size: 初始候选词表大小（越大越精确，越慢）
        em_steps: 每轮剪枝前的 EM 迭代次数

    Returns:
        最终词表 {token: log_prob}
    """
    # Step 1：构建初始大词表
    print(f"构建初始候选词表（最大 {initial_vocab_size} 个 token）...")
    vocab = build_initial_vocab(corpus, max_vocab_size=initial_vocab_size)
    print(f"初始词表大小: {len(vocab)}")

    # 把语料拆成词列表（Viterbi 对词级别操作）
    corpus_words: List[str] = []
    for text in corpus:
        corpus_words.extend(text.strip().split())

    # Step 2-4：EM + 剪枝迭代
    iteration = 0
    while len(vocab) > target_vocab_size:
        iteration += 1
        print(f"\n第 {iteration} 轮迭代（当前词表 {len(vocab)} → 目标 {target_vocab_size}）：")

        # EM 步骤：重新估算概率
        for em_step in range(em_steps):
            expected_counts = compute_token_expected_counts(corpus_words, vocab)
            vocab = update_vocab_probs(expected_counts)
            ll = compute_log_likelihood(corpus_words, vocab)
            print(f"  EM step {em_step + 1}: 对数似然 = {ll:.2f}")

        # 剪枝
        vocab = prune_vocab(vocab, corpus_words, target_size=target_vocab_size)

    print(f"\n训练完成！最终词表大小: {len(vocab)}")
    return vocab


# ═══════════════════════════════════════════════════════════
# STEP 6：Unigram LM 编码和解码
# ═══════════════════════════════════════════════════════════

class UnigramLMTokenizer:
    """
    Unigram LM 分词器。

    推理时使用 Viterbi 算法找最大概率分词方案。

    特色功能：Subword Regularization（子词正则化）
        训练 NMT/LLM 时，不总是用最优分词，而是
        按概率随机采样多种分词方案，增加训练数据多样性。
    """

    def __init__(self, vocab: Dict[str, float], unk_token: str = '<unk>'):
        """
        Args:
            vocab: {token: log_prob}
            unk_token: OOV 用的特殊 token
        """
        self.vocab = vocab
        self.unk_token = unk_token
        # 构建 token2id 和 id2token（按 log_prob 从大到小排序）
        sorted_tokens = sorted(vocab.items(), key=lambda x: -x[1])
        self.token2id: Dict[str, int] = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        self.id2token: Dict[int, str] = {i: t for t, i in self.token2id.items()}
        self.token2id[unk_token] = len(self.token2id)
        self.id2token[len(self.id2token)] = unk_token

    def tokenize(self, text: str) -> List[str]:
        """
        文本 → 最优分词 token 字符串列表。

        对每个词独立做 Viterbi，词间用空格分隔（在 token 上不体现）。
        注意：不同于 BPE，Unigram LM 的词表 token 不区分"词首/词中"，
        模型需要通过上下文理解边界（或用 ▁ 前缀标记词首，SentencePiece 的做法）。
        """
        result: List[str] = []
        for word in text.strip().split():
            tokens, _ = viterbi_segment(word, self.vocab)
            result.extend(tokens)
        return result

    def encode(self, text: str) -> List[int]:
        """文本 → token id 列表（最优分词）"""
        tokens = self.tokenize(text)
        unk_id = self.token2id[self.unk_token]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """token id 列表 → 文本（直接拼接，无法完美还原空格）"""
        # 注意：Unigram LM 没有 </w> 机制，解码时无法区分词边界
        # 实际 SentencePiece 用 ▁ 前缀标记词首来解决这个问题
        tokens = [self.id2token.get(i, self.unk_token) for i in ids]
        return ''.join(tokens)

    def sample_encode(
        self,
        text: str,
        alpha: float = 0.1,
        n_samples: int = 1,
    ) -> List[List[int]]:
        """
        Subword Regularization：按概率随机采样多种分词方案。

        这是 Unigram LM 相对 BPE 的独特优势！
        训练时用多样化分词，提升模型对拼写变体的鲁棒性。

        原理（SampleEncode，Kudo 2018）：
            不用 Viterbi（最优唯一解），而是从所有可能的分词方案
            中按概率采样。alpha 控制多样性：
                alpha → 0：几乎总是选最优分词（等同于 Viterbi）
                alpha → 1：按真实概率采样，多样性最大

        简化实现：对每个词，随机扰动 token 概率后再 Viterbi。

        Args:
            text: 输入文本
            alpha: 平滑参数（0-1，越大多样性越高）
            n_samples: 采样次数

        Returns:
            n_samples 个 token id 列表
        """
        import random

        samples: List[List[int]] = []
        unk_id = self.token2id[self.unk_token]

        for _ in range(n_samples):
            # 给每个 token 的对数概率加随机扰动
            perturbed_vocab = {
                t: p + alpha * math.log(random.random() + 1e-10)
                for t, p in self.vocab.items()
            }
            ids: List[int] = []
            for word in text.strip().split():
                tokens, _ = viterbi_segment(word, perturbed_vocab)
                for t in tokens:
                    ids.append(self.token2id.get(t, unk_id))
            samples.append(ids)

        return samples


# ═══════════════════════════════════════════════════════════
# 主程序演示
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Unigram LM 分词演示（小规模）")
    print("=" * 60)

    # 小语料，快速演示
    corpus = [
        "low lower lowest",
        "new newest newer",
        "wide widest wider",
        "high higher highest",
    ]

    print("\n步骤 1：Viterbi 分词示例（手动构建小词表）")
    # 手动构建一个小词表演示 Viterbi
    small_vocab = {
        'l': math.log(0.1), 'o': math.log(0.08), 'w': math.log(0.09),
        'lo': math.log(0.05), 'low': math.log(0.15), 'low': math.log(0.15),
        'er': math.log(0.08), 'est': math.log(0.06), 'new': math.log(0.12),
        'n': math.log(0.08), 'e': math.log(0.09),
        'wide': math.log(0.10), 'wid': math.log(0.04),
        'high': math.log(0.11), 'hi': math.log(0.05),
        'i': math.log(0.08), 'g': math.log(0.06), 'h': math.log(0.07),
        'a': math.log(0.07), 'd': math.log(0.06), 's': math.log(0.07),
        't': math.log(0.06), 'r': math.log(0.06),
    }

    test_words = ['lowest', 'newer', 'widest', 'highest']
    print("\n Viterbi 最优分词：")
    for word in test_words:
        tokens, score = viterbi_segment(word, small_vocab)
        print(f"  '{word}' → {tokens}  (对数概率 = {score:.3f})")

    print("\n步骤 2：Subword Regularization 随机采样")
    tokenizer = UnigramLMTokenizer(small_vocab)
    print("\n对 'lowest' 采样 5 种分词方案：")
    for i, ids in enumerate(tokenizer.sample_encode("lowest", alpha=0.3, n_samples=5)):
        tokens = [tokenizer.id2token.get(i_, '<unk>') for i_ in ids]
        print(f"  方案 {i + 1}: {tokens}")

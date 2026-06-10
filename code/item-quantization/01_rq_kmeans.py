"""
RQ-KMeans（残差量化 K-Means）完整实现
======================================
将 Item 的连续向量表征（embedding）压缩为多层离散语义 ID（Semantic ID）。
适用于生成式推荐系统，作为 LLM 的 Item Token 表示。

核心思路：
  1. 第 1 层：对所有 Item embedding 做 K-Means，每个 Item 分配到最近的 cluster，
     其 cluster 中心 ID 作为第 1 层 token。
  2. 第 2 层：对"embedding - 第1层 cluster 中心"的残差再做 K-Means。
  3. 以此类推，共做 L 层。
  4. 每个 Item 最终得到 L 个 token，构成 Semantic ID 元组 (s1, s2, ..., sL)。

优点：
  - 轻量，纯 NumPy + sklearn，无需 GPU
  - 可直接作为 RQ-VAE 的初始化码本（热启动效果好）
  - 编码时支持批次处理，内存友好

对应论文：
  - OneRec (arXiv 2502.18965) §2.1 Balanced Identifier Quantization（基础方案）
  - RQ-VAE (arXiv 2203.01941) Residual Quantization 原始方案
  - Semantic-ID (CarterHe479/Semantic-ID on GitHub)

输出格式（与 OneRec 对齐）：
  <|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Optional


# ═══════════════════════════════════════════════════════════
# 核心类：ResidualKMeans
# ═══════════════════════════════════════════════════════════

class ResidualKMeans:
    """
    残差量化 K-Means 分词器。

    将高维连续向量（Item embedding）逐层压缩为离散 token 序列。
    每层用 K-Means 拟合当前残差，将残差映射到最近的 cluster 中心，
    然后把残差减去该中心，得到下一层要拟合的"剩余误差"。

    参数：
        L (int): 量化层数。层数越多，重建精度越高，但 token 序列越长。
                 推荐值：3~4（与 OneRec 保持一致）
        K (int): 每层的 codebook 大小（cluster 数量）。
                 总表达能力 = K^L 种不同组合。
                 推荐值：256（对应 8-bit per layer）
        random_state (int): 随机种子，保证可复现

    属性：
        codebooks (List[np.ndarray]): 训练完成后，每层的 cluster 中心矩阵 [K, d]
    """

    def __init__(self, L: int = 4, K: int = 256, random_state: int = 0):
        self.L = L               # 残差量化层数
        self.K = K               # 每层的 codebook 大小
        self.random_state = random_state
        self.codebooks: List[np.ndarray] = []   # 训练后存放各层 codebook


# ═══════════════════════════════════════════════════════════
# STEP 1：训练 —— 逐层 KMeans 拟合残差
# ═══════════════════════════════════════════════════════════

    def fit(self, Z: np.ndarray) -> "ResidualKMeans":
        """
        训练残差量化 codebook。

        算法流程（以 L=3 为例）：
            第 1 层：
              residual = Z                          ← 初始残差 = 原始 embedding
              KMeans(K).fit(residual)               ← 训练 K 个 cluster
              Q1[i] = cluster_center[assignment[i]] ← 每个 Item 的量化向量
              residual = residual - Q1              ← 更新残差

            第 2 层：
              KMeans(K).fit(residual)               ← 对残差训练新 codebook
              Q2[i] = cluster_center[assignment[i]]
              residual = residual - Q2

            第 3 层：同上...

        最终重建近似：Z ≈ Q1 + Q2 + Q3（各层量化向量的叠加）

        Args:
            Z (np.ndarray): Item embedding 矩阵，形状 [N, d]
                            N = Item 数量，d = embedding 维度

        Returns:
            self（支持链式调用 rqkm.fit(Z).encode(Z)）
        """
        Z = np.asarray(Z, dtype=np.float32)
        residual = Z.copy()          # 初始残差 = 原始 embedding
        self.codebooks = []

        for layer in range(self.L):
            # ---- 对当前残差做 K-Means ----
            km = KMeans(
                n_clusters=self.K,
                n_init="auto",          # sklearn >= 1.2 的推荐写法，自动选初始化次数
                random_state=self.random_state
            ).fit(residual)

            # cluster 中心矩阵：[K, d]，即该层的 codebook
            C = km.cluster_centers_.astype(np.float32, copy=False)

            # 每个 Item 被分配到哪个 cluster（即该层的 token ID）
            idx = km.predict(residual)   # [N]，每个值在 [0, K)

            # 取出该 Item 对应 cluster 中心，作为量化向量
            Q = C[idx]                   # [N, d]

            # 保存当前层 codebook
            self.codebooks.append(C)

            # 更新残差：下一层要拟合"当前层还没覆盖的误差"
            residual = residual - Q

        return self


# ═══════════════════════════════════════════════════════════
# STEP 2：编码 —— 将新的 embedding 映射为 token 序列
# ═══════════════════════════════════════════════════════════

    def encode(self, Z: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """
        将 Item embedding 编码为 L 层离散 token 序列（Semantic ID）。

        与 fit() 不同：encode() 使用已训练好的 codebook，
        不更新任何参数，仅做最近邻查找。

        【低内存优化】
        距离公式展开：
            ||r - c||² = ||r||² - 2·(r · c) + ||c||²

        直接计算 ||r - c||² 需要广播出 [N, K, d] 的三维张量，
        当 N=100万、K=256、d=256 时，显存/内存需要 ~65GB，不现实。

        展开后只需：
          - r_norm: [B, 1]
          - c_norm: [K]
          - r @ C.T: [B, K]  ← 矩阵乘法，无需广播
        总内存 O(B·K)，按批次处理，大幅节省内存。

        Args:
            Z (np.ndarray): 待编码的 embedding 矩阵，形状 [N, d]
            batch_size (int): 每批处理的 Item 数量（内存-速度权衡）

        Returns:
            codes (np.ndarray): Semantic ID 矩阵，形状 [N, L]，dtype=int32
                                codes[i, l] = 第 i 个 Item 在第 l 层的 token ID
        """
        Z = np.asarray(Z, dtype=np.float32)
        N, d = Z.shape
        L = len(self.codebooks)

        # 预分配结果矩阵 [N, L]
        codes = np.empty((N, L), dtype=np.int32)

        # 初始化残差 = 原始 embedding
        residual = Z.copy()

        for l, C in enumerate(self.codebooks):
            # C: [K, d] 当前层的 codebook
            C = C.astype(np.float32, copy=False)
            c_norm = (C * C).sum(axis=1)          # [K]，每个中心向量的模平方

            # 按 batch 分块计算，避免一次性构造 [N, K] 的完整距离矩阵
            for i in range(0, N, batch_size):
                R = residual[i:i + batch_size]                          # [B, d]
                r_norm = (R * R).sum(axis=1, keepdims=True)             # [B, 1]

                # 展开的欧氏距离平方矩阵 [B, K]
                # 公式：||r - c||² = ||r||² - 2·r·c + ||c||²
                dists = r_norm - 2.0 * (R @ C.T) + c_norm[None, :]     # [B, K]

                # 每个 Item 找最近的 cluster（argmin over K 维）
                idx = dists.argmin(axis=1).astype(np.int32)             # [B]

                # 存储该层 token ID
                codes[i:i + batch_size, l] = idx

                # 减去量化向量，更新残差（供下一层使用）
                residual[i:i + batch_size] = R - C[idx]

        return codes


# ═══════════════════════════════════════════════════════════
# STEP 3：保存 / 加载 codebook
# ═══════════════════════════════════════════════════════════

    def save(self, path: str) -> None:
        """
        将训练好的 codebook 保存为 .npz 文件。

        文件结构：
            codebook_0: [K, d]  ← 第 1 层 codebook
            codebook_1: [K, d]  ← 第 2 层 codebook
            ...

        Args:
            path (str): 保存路径，建议以 .npz 结尾
        """
        np.savez(
            path,
            **{f"codebook_{l}": C for l, C in enumerate(self.codebooks)}
        )
        print(f"[RQKMeans] 已保存 {self.L} 层 codebook → {path}")

    def load(self, path: str) -> "ResidualKMeans":
        """
        从 .npz 文件加载 codebook。

        Args:
            path (str): .npz 文件路径

        Returns:
            self（支持链式调用）
        """
        data = np.load(path)
        self.codebooks = [data[f"codebook_{l}"] for l in range(len(data.files))]
        self.L = len(self.codebooks)
        print(f"[RQKMeans] 已加载 {self.L} 层 codebook from {path}")
        return self


# ═══════════════════════════════════════════════════════════
# STEP 4：辅助函数 —— 将 token 序列格式化为 SID 字符串
# ═══════════════════════════════════════════════════════════

def format_sid(codes: np.ndarray, item_idx: int) -> str:
    """
    将单个 Item 的 token 序列格式化为 OneRec 风格的 SID 字符串。

    格式：<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>
    其中 a/b/c 对应层级前缀，c0/c1/c2 是各层的 cluster ID。

    Args:
        codes (np.ndarray): encode() 返回的 [N, L] 矩阵
        item_idx (int): 要格式化的 Item 索引

    Returns:
        str: SID 字符串，可直接插入 LLM prompt
    """
    layer_prefix = "abcdefghijklmnopqrstuvwxyz"   # 最多支持 26 层
    tokens = codes[item_idx]                        # [L]

    inner = "".join(
        f"<s_{layer_prefix[l]}_{tokens[l]}>"
        for l in range(len(tokens))
    )
    return f"<|sid_begin|>{inner}<|sid_end|>"


# ═══════════════════════════════════════════════════════════
# 演示：完整的训练 + 编码 + 格式化流程
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("RQ-KMeans 演示：Item Embedding → Semantic ID")
    print("=" * 60)

    # 模拟 500 个 Item，embedding 维度 = 64
    np.random.seed(42)
    N, d = 500, 64
    embeddings = np.random.randn(N, d).astype(np.float32)

    # ---- 训练 ----
    print(f"\n[1] 训练 RQ-KMeans（L=3 层，每层 K=32 个 cluster）...")
    rqkm = ResidualKMeans(L=3, K=32, random_state=0)
    rqkm.fit(embeddings)
    print(f"    训练完成，codebook 形状：{[C.shape for C in rqkm.codebooks]}")

    # ---- 编码 ----
    print(f"\n[2] 编码全量 Item embedding...")
    codes = rqkm.encode(embeddings, batch_size=128)
    print(f"    编码结果形状：{codes.shape}")   # (500, 3)
    print(f"    前 5 个 Item 的 codes：\n    {codes[:5]}")

    # ---- 格式化 ----
    print(f"\n[3] 格式化为 SID 字符串：")
    for i in range(3):
        print(f"    Item {i}: {format_sid(codes, i)}")

    # ---- 重建误差 ----
    print(f"\n[4] 验证重建质量...")
    # 重建：对每个 Item 把各层 cluster 中心相加
    reconstructed = np.zeros_like(embeddings)
    for l, C in enumerate(rqkm.codebooks):
        reconstructed += C[codes[:, l]]
    mse = np.mean((embeddings - reconstructed) ** 2)
    print(f"    重建 MSE = {mse:.6f}（层数越多误差越小）")

    # ---- 保存 / 加载测试 ----
    print(f"\n[5] 保存 + 加载 codebook...")
    rqkm.save("/tmp/rqkmeans_codebook.npz")
    rqkm2 = ResidualKMeans(L=3, K=32).load("/tmp/rqkmeans_codebook.npz")
    codes2 = rqkm2.encode(embeddings[:10])
    assert np.array_equal(codes[:10], codes2), "加载后编码结果不一致！"
    print("    加载后编码结果与原始一致 ✓")

    print("\n演示完成！")


# ═══════════════════════════════════════════════════════════
# ASCII 数据流图：RQ-KMeans 完整执行过程
# ═══════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  阶段 A：fit(Z) —— 训练，逐层 KMeans 拟合残差                    │
# │                                                                  │
# │   Z [N,d]  ──────────────────────────────────────────────────┐  │
# │   (全量 embedding)                                            │  │
# │                                                               │  │
# │   residual = Z.copy()    ← 初始残差 = 原始 embedding [N,d]    │  │
# │       │                                                       │  │
# │       ▼                                                       │  │
# │  ╔══════════════════════════════════════════════════╗         │  │
# │  ║ Layer 1                                          ║         │  │
# │  ║                                                  ║         │  │
# │  ║   KMeans(K).fit(residual)                        ║         │  │
# │  ║         │                                        ║         │  │
# │  ║         ├─→ codebook[0] [K,d]   ← 保存第1层码本  ║         │  │
# │  ║         │                                        ║         │  │
# │  ║         └─→ idx [N]  = 每个点最近 cluster 编号   ║         │  │
# │  ║                │                                 ║         │  │
# │  ║   Q1 [N,d] = codebook[0][idx]  ← 量化向量        ║         │  │
# │  ║                │                                 ║         │  │
# │  ║   residual = residual - Q1      ← 更新残差 [N,d] ║         │  │
# │  ╚══════════════════════════════════════════════════╝         │  │
# │       │                                                       │  │
# │       ▼  (residual 现在是 Z 被第1层"解释"后剩余的误差)          │  │
# │  ╔══════════════════════════════════════════════════╗         │  │
# │  ║ Layer 2                                          ║         │  │
# │  ║                                                  ║         │  │
# │  ║   KMeans(K).fit(residual)   ← 拟合第1层的残差    ║         │  │
# │  ║         ├─→ codebook[1] [K,d]                   ║         │  │
# │  ║         └─→ idx [N]                             ║         │  │
# │  ║   Q2 = codebook[1][idx]                         ║         │  │
# │  ║   residual = residual - Q2                      ║         │  │
# │  ╚══════════════════════════════════════════════════╝         │  │
# │       │                                                       │  │
# │       ▼  (重复 L 层，每层残差越来越小)                          │  │
# │  ╔══════════════════════════════════════════════════╗         │  │
# │  ║ Layer L                                          ║         │  │
# │  ║   KMeans(K).fit(residual)                        ║         │  │
# │  ║         └─→ codebook[L-1] [K,d]                 ║         │  │
# │  ╚══════════════════════════════════════════════════╝         │  │
# │                                                               │  │
# │   训练结果：codebooks = [C0,C1,...,C_{L-1}]，每个 [K,d]       │  │
# └───────────────────────────────────────────────────────────────┘  │
#                                                                     │
# ┌──────────────────────────────────────────────────────────────────┤
# │  阶段 B：encode(Z) —— 推理，低内存分批最近邻编码                  │
# │                                                                  │
# │   Z [N,d]  ──→  residual = Z.copy()                             │
# │   codes = empty [N, L]                                          │
# │                                                                  │
# │   for l in range(L):                 ← 逐层处理                  │
# │     C = codebook[l]  [K,d]                                      │
# │     c_norm = (C*C).sum(axis=1)  [K]  ← 预算 ||c||²              │
# │                                                                  │
# │     for batch i in range(0, N, B):   ← 分 batch 节省内存        │
# │       R = residual[i:i+B]  [B,d]                                │
# │       r_norm = (R*R).sum(axis=1, keepdim)  [B,1]                │
# │                                                                  │
# │       ┌─ 距离展开（避免 [B,K,d] 广播）─────────────────────┐    │
# │       │  dist[B,K] = r_norm - 2*(R @ C.T) + c_norm        │    │
# │       │            = ||R||² - 2·R·C^T + ||C||²            │    │
# │       │            ≡ ||R - C||²  （欧氏距离平方）           │    │
# │       └───────────────────────────────────────────────────┘    │
# │                                                                  │
# │       idx[B]  = dist.argmin(axis=1)  ← 每个样本最近的 cluster   │
# │       codes[i:i+B, l] = idx                                     │
# │       residual[i:i+B] -= C[idx]      ← 减去量化向量，准备下层   │
# │                                                                  │
# │   返回 codes [N, L]：第 i 行 = Item i 的 Semantic ID 元组       │
# │                                                                  │
# │   例：codes[42] = [5, 23, 7]  表示                              │
# │     → 第1层选了 cluster 5，第2层选了 cluster 23，第3层选了 7     │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  阶段 C：format_sid() —— 格式化为 LLM prompt 中的 token 串       │
# │                                                                  │
# │   codes[42] = [5, 23, 7]                                        │
# │       │                                                          │
# │       ▼                                                          │
# │   "<|sid_begin|><s_a_5><s_b_23><s_c_7><|sid_end|>"             │
# │        ↑                 ↑              ↑                        │
# │     层级前缀 a/b/c/...  cluster ID    结束标记                   │
# │                                                                  │
# │   直接插入 LLM 的 user/assistant 消息，模型把这串当作 Item token  │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  重建验证（理解残差量化本质）                                      │
# │                                                                  │
# │   Z ≈ C0[s1] + C1[s2] + C2[s3] + ... + C_{L-1}[sL]            │
# │        ↑        ↑        ↑                                       │
# │     第1层   第2层拟合  第3层拟合                                  │
# │     最粗的  第1层残差  第2层残差                                  │
# │     近似                                                         │
# │                                                                  │
# │   → 层数越多，重建 MSE 越小，但 SID 越长                         │
# │   → L=4, K=256：可区分 256^4 ≈ 4B 种不同 Item                   │
# └──────────────────────────────────────────────────────────────────┘
#
# 类方法速查：
#   ResidualKMeans
#     fit(Z)               ← 训练：逐层 KMeans 拟合残差
#     encode(Z)            ← 推理：分批最近邻，返回 codes [N,L]
#     save(path)           ← 保存所有层 codebook 到 .npz
#     load(path)           ← 从 .npz 加载
#     format_sid(codes, i) ← 辅助：将 codes[i] 格式化为 SID 字符串

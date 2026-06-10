"""
VQ-VAE（向量量化变分自编码器）完整实现
========================================
最经典的离散表征学习方法，只用一个码本（codebook）将 embedding 映射为单个 token。
与 RQ-VAE / RQ-KMeans 相比：
  - RQ 系列：L 层残差量化 → 每个 Item 得到 L 个 token（分层精细表示）
  - VQ-VAE：单层量化 → 每个 Item 得到 1 个 token（简单紧凑，但需要大码本）

为了区分同样数量的 Item，码本大小 K 需要 ≥ Item 数量的数量级。
例如：100 万 Item → K 至少需要 4096~32768（依赖 cluster 内 Item 数可接受）。

核心组件：
  1. EMA（指数移动平均）更新码本：比反传梯度更稳定，无需额外 optimizer
  2. K 维分块距离计算：避免 [B, K, d] 大张量，显存友好
  3. Dead Code Restart：对长期不被使用的 cluster，用 batch 中的随机样本替换，
     防止码本利用率低下（collapse）

STE 说明（与 RQ-VAE 相同）：
  量化不可微 → 用 z_q_ste = z_e + (z_q - z_e).detach() 让梯度直通

对应论文：
  - VQ-VAE (arXiv 1711.00937)："Neural Discrete Representation Learning"，van den Oord et al.
  - VQ-VAE-2 (arXiv 1906.00446)：层次化 VQ-VAE
  - Semantic-ID (CarterHe479/Semantic-ID)：用于生成式推荐的 VQ-VAE 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ═══════════════════════════════════════════════════════════
# 核心类：VectorQuantizerEMA
# ═══════════════════════════════════════════════════════════

class VectorQuantizerEMA(nn.Module):
    """
    低内存 VQ-VAE 量化器（EMA 更新 + 分块距离计算）。

    参数：
        K (int): 码本大小（cluster 数量）。
                 单层量化时需较大（4096 ~ 32768），以区分足够多的 Item。
        d (int): embedding 维度
        beta (float): Commitment Loss 权重（典型值 0.25）
        decay (float): EMA 衰减系数（典型值 0.99）
        eps (float): Laplace 平滑，防止除以零
        k_chunk (int): 分块大小，越小显存越低（推荐 512~2048）
        restart_unused_codes (bool): 是否开启 Dead Code Restart
        restart_threshold (float): ema_cluster_size 低于此阈值则认为是"死亡 cluster"
    """

    def __init__(
        self,
        K: int = 4096,
        d: int = 256,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        k_chunk: int = 1024,
        restart_unused_codes: bool = False,
        restart_threshold: float = 0.25,
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.k_chunk = max(1, int(k_chunk))
        self.restart_unused_codes = bool(restart_unused_codes)
        self.restart_threshold = float(restart_threshold)

        # 码本矩阵：[K, d]，作为 nn.Parameter，但通过 EMA 更新（不走 optimizer）
        self.codebook = nn.Parameter(torch.randn(K, d))

        # EMA buffers（随 .to(device) 自动迁移）
        # ema_cluster_size[k]：第 k 个 cluster 的 EMA 使用频率（样本数的移动平均）
        self.register_buffer("ema_cluster_size", torch.ones(K))
        # ema_weight[k, :]：第 k 个 cluster 的 EMA 累积向量和
        self.register_buffer("ema_weight", self.codebook.detach().clone())


# ═══════════════════════════════════════════════════════════
# STEP 1：外部初始化码本（可选）
# ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def init_codebook(self, codebook: torch.Tensor) -> None:
        """
        用外部码本初始化（例如用 MiniBatchKMeans 预训练的结果）。

        sklearn 的 MiniBatchKMeans 可以快速给出好的初始 codebook：
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=K).fit(embeddings)
            init_cb = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            vq.init_codebook(init_cb)

        Args:
            codebook (Tensor): 形状 [K, d] 的初始码本
        """
        codebook = codebook.to(device=self.codebook.device, dtype=self.codebook.dtype)
        if tuple(codebook.shape) != tuple(self.codebook.shape):
            raise ValueError(
                f"期望码本形状 {tuple(self.codebook.shape)}，"
                f"实际得到 {tuple(codebook.shape)}"
            )
        self.codebook.copy_(codebook)
        # 重置 EMA buffer，避免旧统计信息污染新码本
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight.copy_(codebook)
        print(f"[VQ-VAE] 已初始化码本，形状 {tuple(codebook.shape)}")


# ═══════════════════════════════════════════════════════════
# STEP 2：低内存最近邻查找（K 维分块）
# ═══════════════════════════════════════════════════════════

    def _argmin_chunked(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 codebook K 维上分块查找最近向量，避免 [B, K, d] 广播。

        距离展开：||z_e - c||² = ||z_e||² - 2·z_e·c^T + ||c||²
        每次只计算 [B, k_chunk] 的局部距离矩阵，再取全局最小。

        Args:
            z_e (Tensor): encoder 输出，形状 [B, d]

        Returns:
            idx_min (Tensor): 最近 cluster 索引，形状 [B]
            dist_min (Tensor): 最小距离值，形状 [B]（用于 Dead Code Restart 判断）
        """
        B, d = z_e.shape
        e_norm = (z_e * z_e).sum(1, keepdim=True)   # [B, 1]，||z_e||² per sample
        idx_min, dist_min = None, None
        K = self.codebook.shape[0]

        for start in range(0, K, self.k_chunk):
            end = min(start + self.k_chunk, K)
            C = self.codebook[start:end]                              # [Ck, d]
            c_norm = (C * C).sum(1)                                   # [Ck]
            dists = e_norm - 2.0 * (z_e @ C.t()) + c_norm[None, :]   # [B, Ck]

            d_chunk, i_chunk = torch.min(dists, dim=1)                # [B]
            i_chunk = i_chunk + start                                 # 修正为全局索引

            if dist_min is None:
                dist_min, idx_min = d_chunk, i_chunk
            else:
                mask = d_chunk < dist_min
                dist_min = torch.where(mask, d_chunk, dist_min)
                idx_min = torch.where(mask, i_chunk, idx_min)

        return idx_min, dist_min


# ═══════════════════════════════════════════════════════════
# STEP 3：前向传播（核心逻辑）
# ═══════════════════════════════════════════════════════════

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VQ-VAE 前向传播。

        训练流程：
          1. 找到距离 z_e 最近的 codebook 向量 z_q = codebook[idx]
          2. EMA 更新 codebook（不走 optimizer）
          3. （可选）Dead Code Restart：替换长期不被使用的 cluster
          4. STE：让梯度可以从 z_q 直通回 z_e
          5. 计算 commitment loss

        推理流程：
          - 仅执行步骤 1，找最近的 code，返回 idx

        Args:
            z_e (Tensor): encoder 输出，形状 [B, d]

        Returns:
            z_q_ste (Tensor): STE 量化向量，形状 [B, d]（可参与梯度计算）
            idx (Tensor): 量化 token ID，形状 [B]（即 Semantic ID）
            commit (Tensor): Commitment Loss 标量
        """
        z_e = z_e.float()

        # ---- 步骤 1：找最近的 cluster ----
        idx, _ = self._argmin_chunked(z_e)             # idx: [B]
        z_q = self.codebook.index_select(0, idx)        # z_q: [B, d]

        # ---- 步骤 2：EMA 更新码本（仅训练期）----
        if self.training:
            with torch.no_grad():
                decay, eps = self.decay, self.eps

                # 统计本 batch 各 cluster 的使用次数
                counts = torch.bincount(idx, minlength=self.K).to(z_e.dtype)  # [K]

                # EMA 更新使用频率：size_k ← decay * size_k + (1-decay) * count_k
                self.ema_cluster_size.mul_(decay).add_(counts, alpha=1 - decay)

                # EMA 更新向量和：weight_k ← decay * weight_k + (1-decay) * Σ_{i→k} z_e_i
                dw = torch.zeros_like(self.ema_weight)      # [K, d]
                dw.index_add_(0, idx, z_e)                  # 累加分配到该 cluster 的 z_e
                self.ema_weight.mul_(decay).add_(dw, alpha=1 - decay)

                # 计算新中心：center_k = weight_k / size_k（Laplace 平滑防除零）
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + eps) / (n + self.K * eps) * n
                )                                           # [K]，平滑后频率
                new_codebook = self.ema_weight / cluster_size.unsqueeze(1)  # [K, d]
                self.codebook.copy_(new_codebook)

                # ---- 步骤 3：Dead Code Restart ----
                # 问题：某些 cluster 长期没有任何 Item 分配（dead code），
                #       这些 cluster 的向量不再更新，白白浪费码本容量。
                # 解决：把这些 dead cluster 的向量替换为 batch 中的随机样本，
                #       相当于"重新激活"该 cluster，让它从一个真实数据点出发重新聚类。
                if self.restart_unused_codes and z_e.shape[0] > 0:
                    # dead cluster：本 batch 没被选中 且 EMA 频率低于阈值
                    dead = (counts == 0) & (self.ema_cluster_size < self.restart_threshold)
                    n_dead = int(dead.sum().item())

                    if n_dead > 0:
                        # 从 batch 中随机采样 n_dead 个样本替换 dead cluster
                        repl_idx = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
                        repl = z_e.index_select(0, repl_idx)          # [n_dead, d]
                        self.codebook[dead].copy_(repl)                # 替换 codebook
                        self.ema_weight[dead].copy_(repl)              # 重置 EMA weight
                        self.ema_cluster_size[dead].fill_(1.0)         # 重置 EMA size

        # ---- 步骤 4：STE（直通估计器）----
        # 前向：使用量化后的 z_q 参与后续计算
        # 反向：梯度从 z_q 直接流向 z_e（绕过 argmin）
        z_q_ste = z_e + (z_q - z_e).detach()

        # ---- 步骤 5：Commitment Loss ----
        # 鼓励 encoder 输出的 z_e 靠近被选中的 cluster 中心 z_q
        # 注意：z_q 用 .detach() 断开梯度，只让 z_e 靠近 z_q，而不是反过来
        # （codebook 已由 EMA 更新，不需要也不应该通过此 loss 更新）
        commit = self.beta * F.mse_loss(z_e, z_q.detach())

        return z_q_ste, idx, commit


# ═══════════════════════════════════════════════════════════
# STEP 4：编码（推理时批量获取 token ID）
# ═══════════════════════════════════════════════════════════

    def encode(self, Z: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        """
        推理阶段：批量将 embedding 编码为单层 token ID（不更新 codebook）。

        Args:
            Z (Tensor): Item embedding，形状 [N, d]
            batch_size (int): 每批处理的 Item 数量

        Returns:
            np.ndarray: token ID 数组，形状 [N]，dtype=int32
        """
        self.eval()
        N = Z.shape[0]
        all_ids = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = Z[i:i + batch_size].float()
                _, idx, _ = self.forward(batch)
                all_ids.append(idx.cpu().numpy())

        return np.concatenate(all_ids, axis=0).astype(np.int32)

    def get_codebook_stats(self) -> dict:
        """
        统计码本的健康状态，用于诊断 collapse。

        Returns:
            dict: {
                "unique_used": 在某次 encode 后唯一使用的 cluster 数，
                "ema_entropy": EMA 频率分布的熵（越大越均匀，最大值 log(K)），
                "max_cluster_size": EMA 中使用频率最高的 cluster 的大小，
                "min_cluster_size": 最低使用频率 cluster 的大小
            }
        """
        size = self.ema_cluster_size.cpu()
        p = size / size.sum().clamp_min(1e-9)
        ent = -(p * p.clamp_min(1e-30).log()).sum().item()
        return {
            "ema_entropy": ent,
            "max_ema_log_K": np.log(self.K),       # 理论最大熵（均匀分布）
            "max_cluster_size": size.max().item(),
            "min_cluster_size": size.min().item(),
        }


# ═══════════════════════════════════════════════════════════
# 演示：完整训练 + 评估流程
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    import numpy as np

    print("=" * 60)
    print("VQ-VAE 演示：Item Embedding → 单层离散 Token")
    print("=" * 60)

    # 设备选择
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[设备] 使用：{device}")

    # 模拟数据：500 个 Item，embedding 维度 = 32
    torch.manual_seed(42)
    N, d = 500, 32
    embeddings = torch.randn(N, d).to(device)

    # ---- 初始化 VQ-VAE ----
    vq = VectorQuantizerEMA(
        K=64,                        # 码本大小（64 个 cluster）
        d=d,                         # embedding 维度
        beta=0.25,                   # commitment loss 权重
        decay=0.99,                  # EMA 衰减
        k_chunk=16,                  # 分块大小（演示用，实际可设 512）
        restart_unused_codes=True,   # 开启 Dead Code Restart
        restart_threshold=0.25,      # EMA 频率低于 0.25 的 cluster 被视为死亡
    ).to(device)

    print(f"\n[1] 模型参数：K={vq.K}, d={vq.d}, 码本参数量={vq.K * vq.d}")

    # ---- 用 MiniBatchKMeans 初始化码本（可选，效果更好）----
    print(f"\n[2] 用 MiniBatchKMeans 初始化码本...")
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=vq.K, random_state=42, n_init="auto")
        km.fit(embeddings.cpu().numpy())
        init_cb = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        vq.init_codebook(init_cb.to(device))
    except Exception as e:
        print(f"    (跳过，原因: {e}，使用随机初始化)")

    # ---- 模拟训练 ----
    print(f"\n[3] 模拟训练（5 个 epoch，batch_size=64）...")
    vq.train()
    batch_size = 64
    for epoch in range(5):
        total_commit = 0.0
        dead_restarts = 0
        n_batch = 0

        for i in range(0, N, batch_size):
            batch = embeddings[i:i + batch_size]
            z_q_ste, idx, commit = vq(batch)
            total_commit += commit.item()
            n_batch += 1

        print(f"    Epoch {epoch+1}: commit_loss={total_commit/n_batch:.6f}")

    # ---- 推理：编码全量 Item ----
    print(f"\n[4] 推理：编码 {N} 个 Item...")
    codes = vq.encode(embeddings)
    print(f"    codes 形状：{codes.shape}")       # (500,)
    print(f"    前 10 个 token ID：{codes[:10]}")

    # ---- 码本健康度统计 ----
    print(f"\n[5] 码本健康度统计：")
    stats = vq.get_codebook_stats()
    unique_used = len(np.unique(codes))
    print(f"    使用的 cluster 数：{unique_used} / {vq.K}")
    print(f"    EMA 熵：{stats['ema_entropy']:.4f}（理论最大 {stats['max_ema_log_K']:.4f}）")
    print(f"    均匀度 = 熵/最大熵 = {stats['ema_entropy']/stats['max_ema_log_K']:.3f}（越接近 1 越均匀）")

    # ---- 与 RQ-KMeans 的区别说明 ----
    print(f"\n[6] VQ vs RQ 区别：")
    print(f"    VQ-VAE：1 层 × K={vq.K} → 每个 Item 1 个 token，可区分 {vq.K} 种")
    print(f"    RQ 系列：L=3 层 × K=32 → 每个 Item 3 个 token，可区分 {32**3} 种")
    print(f"    → 要达到同等区分能力，VQ 需要 K={32**3}（内存更大），RQ 用 3 个较小码本叠加")

    print("\n演示完成！")


# ═══════════════════════════════════════════════════════════
# ASCII 数据流图：VQ-VAE 完整执行过程
# ═══════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  forward(z_e) —— 单次 batch 的完整流程（单层量化）                │
# │                                                                  │
# │  z_e [B,d]  ← encoder 输出（item embedding）                     │
# │      │                                                           │
# │      ▼                                                           │
# │  ╔═══ 步骤1：分块最近邻 _argmin_chunked(z_e) ════════════════╗   │
# │  ║                                                            ║   │
# │  ║  e_norm[B,1] = ||z_e||²   ← 预算每个样本的模平方           ║   │
# │  ║                                                            ║   │
# │  ║  for chunk in range(0, K, k_chunk):  ← 按 K 维分块         ║   │
# │  ║    C = codebook[chunk : chunk+k_chunk]  [Ck, d]            ║   │
# │  ║    c_norm[Ck]  = ||C||²                                    ║   │
# │  ║    dist[B,Ck]  = e_norm - 2*(z_e @ C.T) + c_norm          ║   │
# │  ║               ≡ ||z_e - C||²  (欧氏距离平方，展开避免广播) ║   │
# │  ║    取 chunk 内最小 → 与全局最小比较 → 更新全局 idx, dist    ║   │
# │  ║                                                            ║   │
# │  ║  idx[B]   = 全局 argmin → 每个样本最近的 codebook 行号      ║   │
# │  ║  z_q[B,d] = codebook[idx]  ← 量化向量                      ║   │
# │  ╚════════════════════════════════════════════════════════════╝   │
# │      │                                                           │
# │      ▼                                                           │
# │  ╔═══ 步骤2：EMA 更新码本（training=True 时执行）══════════════╗   │
# │  ║                                                            ║   │
# │  ║  counts[K] = bincount(idx, minlength=K)                    ║   │
# │  ║           → 本 batch 中第 k 个 cluster 被选中几次           ║   │
# │  ║                                                            ║   │
# │  ║  ema_cluster_size ← 0.99 * ema_size + 0.01 * counts       ║   │
# │  ║  (追踪每个 cluster 的"长期平均使用频率")                     ║   │
# │  ║                                                            ║   │
# │  ║  dw[K,d] = 0                                              ║   │
# │  ║  dw.index_add_(0, idx, z_e)  ← 累加被分配到 cluster k 的   ║   │
# │  ║                                  所有 z_e 向量之和          ║   │
# │  ║  ema_weight ← 0.99 * ema_weight + 0.01 * dw               ║   │
# │  ║                                                            ║   │
# │  ║  Laplace 平滑（防 size=0 除零）：                           ║   │
# │  ║    n = ema_size.sum()                                      ║   │
# │  ║    smooth_size = (ema_size+eps) / (n+K*eps) * n            ║   │
# │  ║                                                            ║   │
# │  ║  new_codebook = ema_weight / smooth_size  [K,d]            ║   │
# │  ║  codebook.copy_(new_codebook)  ← 原地更新，不走 optimizer   ║   │
# │  ║                                                            ║   │
# │  ╠═══ 步骤2b：Dead Code Restart（restart_unused_codes=True）══╣   │
# │  ║                                                            ║   │
# │  ║  dead[K] = (counts == 0) & (ema_size < threshold)         ║   │
# │  ║         → 本 batch 没被选中 且 长期频率低                   ║   │
# │  ║                                                            ║   │
# │  ║  if n_dead > 0:                                            ║   │
# │  ║    repl_idx = randint(0, B, size=n_dead)                   ║   │
# │  ║    codebook[dead] = z_e[repl_idx]  ← 用真实数据点替换      ║   │
# │  ║    ema_weight[dead] = z_e[repl_idx]                        ║   │
# │  ║    ema_size[dead]   = 1.0           ← 重置 EMA 统计         ║   │
# │  ║    (效果：cluster 从数据点重新出发，有机会被激活)             ║   │
# │  ╚════════════════════════════════════════════════════════════╝   │
# │      │                                                           │
# │      ▼                                                           │
# │  ╔═══ 步骤3：STE（Straight-Through Estimator 直通估计器）═════╗   │
# │  ║                                                            ║   │
# │  ║  z_q_ste = z_e + (z_q - z_e).detach()                     ║   │
# │  ║                                                            ║   │
# │  ║  为什么需要 STE？                                           ║   │
# │  ║    argmin 操作不可微 → 梯度在量化处断掉                     ║   │
# │  ║    encoder 无法通过 z_q 得到梯度                            ║   │
# │  ║                                                            ║   │
# │  ║  STE 的数学技巧：                                           ║   │
# │  ║    前向：z_q_ste 的值 = z_q（量化后的向量）                 ║   │
# │  ║    反向：∂loss/∂z_e = ∂loss/∂z_q_ste * 1  （梯度直通）     ║   │
# │  ║    实现：(z_q - z_e).detach() → 这一段对 z_e 的梯度为 0    ║   │
# │  ║           z_e 的部分梯度来自外部 loss，正常反传             ║   │
# │  ╚════════════════════════════════════════════════════════════╝   │
# │      │                                                           │
# │      ▼                                                           │
# │  ╔═══ 步骤4：Commitment Loss ══════════════════════════════════╗   │
# │  ║                                                            ║   │
# │  ║  commit = β * MSE(z_e, z_q.detach())                      ║   │
# │  ║                                                            ║   │
# │  ║  含义：让 encoder 输出 z_e 靠近 codebook 中心 z_q          ║   │
# │  ║       z_q.detach() → codebook 不通过此 loss 更新           ║   │
# │  ║       （codebook 只通过 EMA 更新，两条路径完全解耦）         ║   │
# │  ╚════════════════════════════════════════════════════════════╝   │
# │      │                                                           │
# │      ▼                                                           │
# │  返回：(z_q_ste [B,d], idx [B], commit_loss)                    │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  encode(Z) —— 推理时批量获取 token ID（eval 模式）                │
# │                                                                  │
# │  model.eval()   ← 关闭 training 标志，EMA 不更新                  │
# │                                                                  │
# │  for i in range(0, N, batch_size):                               │
# │    batch = Z[i:i+batch_size]  [B, d]                             │
# │    _, idx, _ = model.forward(batch)                              │
# │    all_ids.append(idx.cpu().numpy())                             │
# │                                                                  │
# │  返回 codes [N]  ← 每个 item 对应 1 个 token ID（单层 VQ）        │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  Dead Code 问题 & Restart 解决思路                                │
# │                                                                  │
# │  数据空间示意（2D）：                                             │
# │                                                                  │
# │    ·····  ← 数据密集区（cluster A、B、C 聚在这里）               │
# │    · · ·                                                         │
# │    ·····           × ← dead cluster D（初始化在稀疏区）          │
# │                                                                  │
# │  问题：cluster D 从未被选中 → ema_weight 不更新 → 永远是死码      │
# │                                                                  │
# │  Restart：把 cluster D 的中心挪到某个真实 item 位置               │
# │    → D 现在处于数据分布内部                                       │
# │    → 下一个 batch 有机会被选中                                    │
# │    → EMA 开始追踪周边 item，cluster D 被"激活"                    │
# └──────────────────────────────────────────────────────────────────┘
#
# 类方法速查：
#   VectorQuantizerEMA(nn.Module)
#     init_codebook(codebook)    ← 外部初始化（如 MiniBatchKMeans）
#     _argmin_chunked(z_e)       ← 内部：分块最近邻，返回 idx, dist
#     forward(z_e)               ← 训练+推理主流程（含 EMA + Restart + STE）
#     encode(Z, batch_size)      ← 推理批量编码 → codes [N]
#     get_codebook_stats()       ← 诊断 collapse：熵、最大/最小 cluster 大小
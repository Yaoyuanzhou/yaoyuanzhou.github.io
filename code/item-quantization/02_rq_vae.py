"""
RQ-VAE（残差向量量化 + EMA 更新）完整实现
==========================================
在 RQ-KMeans 的基础上引入：
  1. EMA（指数移动平均）更新 codebook，无需梯度 / optimizer，训练稳定
  2. 批内熵正则（Entropy Regularization），防止 codebook collapse（少数 cluster 垄断）
  3. K 维分块计算距离，显存友好，适配 macOS MPS / CUDA

与 RQ-KMeans 的关系：
  - 量化结构完全相同（残差逐层量化）
  - 差异在于 codebook 的更新方式：
      RQ-KMeans: sklearn KMeans 批量 EM（一次性处理全量数据）
      RQ-VAE:    EMA 在线更新（每个 batch 更新一次，适合大规模数据 / 迭代优化）
  - 实践推荐：先用 RQ-KMeans 训好 codebook，再用 init_from_rqkmeans() 热启动
    RQ-VAE，效果明显好于随机初始化。

输出（forward 返回）：
  - z_q_total: 量化后的向量（STE 直通），可参与梯度计算
  - ids: 各层 token ID 列表，len = L，每个元素形状 [B]
  - commit: Commitment Loss（鼓励 encoder 输出贴近 codebook）
  - loss_reg: 熵正则 loss（鼓励 codebook 均匀被使用）

对应论文：
  - RQ-VAE (arXiv 2203.01941)："Autoregressive Image Generation using Residual Quantization"
  - OneRec (arXiv 2502.18965) §2.1：使用此方案的变体
  - Semantic-ID (CarterHe479/Semantic-ID)

STE（Straight-Through Estimator，直通估计器）说明：
  量化操作（argmin）不可微，梯度无法反传。
  STE 的做法：
    前向：用量化后的向量 z_q 参与计算
    反向：梯度"直接跳过"量化，把 z_q 的梯度复制给 z_e
  实现：z_q_ste = z_e + (z_q - z_e).detach()
        → 前向值 = z_q，但对 z_e 的导数 = 1（梯度直通）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════
# 核心类：ResidualQuantizer（RQ-VAE）
# ═══════════════════════════════════════════════════════════

class ResidualQuantizer(nn.Module):
    """
    RQ-VAE 残差量化器（低内存 + EMA 版）。

    参数：
        L (int): 量化层数，每层对应一个独立 codebook
        K (int): 每层 codebook 大小（cluster 数量）
        d (int): embedding 维度
        beta (float): Commitment Loss 权重
                      loss_commit = beta * MSE(z_e, z_q.detach())
                      作用：让 encoder 输出的 z_e 贴近 codebook，而非让 codebook 追逐 z_e
        entropy_reg (float): 熵正则权重。
                              > 0 时对"使用频率过高的 cluster"施加距离惩罚，
                              使各 cluster 的使用率趋于均匀，避免 codebook collapse。
                              设为 0 则关闭此正则。
        k_chunk (int): 计算距离时 codebook K 维度的分块大小。
                       分块越小显存占用越低，但稍慢。推荐 512~2048。
        ema_decay (float): EMA 衰减系数（momentum）。
                           0.99 表示"旧值权重 99%，新值权重 1%"，更新平滑。
        eps (float): Laplace 平滑项，防止 cluster 大小为 0 时除以零
    """

    def __init__(
        self,
        L: int = 4,
        K: int = 256,
        d: int = 256,
        beta: float = 0.25,
        entropy_reg: float = 1e-4,
        k_chunk: int = 1024,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.d = d
        self.beta = beta
        self.entropy_reg = entropy_reg
        self.k_chunk = max(1, int(k_chunk))
        self.ema_decay = ema_decay
        self.eps = eps

        # 各层 codebook，形状 [K, d]，作为可训练参数（但实际由 EMA 更新，不走 optimizer）
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(K, d)) for _ in range(L)]
        )

        # EMA buffers（不参与梯度计算，register_buffer 保证跟随 .to(device) 移动）
        # ema_cluster_size[l, k]：第 l 层第 k 个 cluster 的 EMA 使用频率
        self.register_buffer("ema_cluster_size", torch.ones(L, K))
        # ema_weight[l, k, :]：第 l 层第 k 个 cluster 的 EMA 累积向量和（除以 size 得中心）
        self.register_buffer(
            "ema_weight",
            torch.stack([p.detach().clone() for p in self.codebooks], dim=0)
        )


# ═══════════════════════════════════════════════════════════
# STEP 1：从 RQ-KMeans 初始化（热启动）
# ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def init_from_rqkmeans(self, codebooks_np: np.ndarray) -> None:
        """
        用 RQ-KMeans 训练好的 codebook 初始化 RQ-VAE。

        推荐做法：
          1. 先用 RQ-KMeans.fit(train_emb) 快速训练 codebook
          2. 用此函数将 codebook 复制到 RQ-VAE
          3. 再用 RQ-VAE.train() 做在线精炼（效果比随机初始化好很多）

        Args:
            codebooks_np (np.ndarray): 形状 [L, K, d] 的 numpy 数组
                                       通常来自 np.load(path)["codebook_0"] 等
        """
        if tuple(codebooks_np.shape) != (self.L, self.K, self.d):
            raise ValueError(
                f"期望 codebook 形状 {(self.L, self.K, self.d)}，"
                f"实际得到 {tuple(codebooks_np.shape)}"
            )
        # 将每层 numpy codebook 复制到 nn.Parameter
        for param, C in zip(self.codebooks, codebooks_np):
            param.copy_(torch.tensor(C, dtype=param.dtype, device=param.device))

        # 重置 EMA buffer（避免残留旧统计信息干扰新 codebook）
        self.ema_cluster_size.fill_(1.0)
        self.ema_weight.copy_(
            torch.stack([p.data for p in self.codebooks], dim=0)
        )
        print(f"[RQ-VAE] 已从 RQ-KMeans codebook 初始化（形状 {codebooks_np.shape}）")


# ═══════════════════════════════════════════════════════════
# STEP 2：低内存最近邻查找（K 维分块）
# ═══════════════════════════════════════════════════════════

    def _argmin_chunked(
        self,
        R: torch.Tensor,
        C: torch.Tensor,
        distance_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 K 维上分块查找最近 cluster，避免构造 [B, K, d] 广播张量。

        距离展开：||R - C||² = ||R||² - 2·R·C^T + ||C||²

        分块策略：每次只取 Codebook 的 k_chunk 列计算一个局部距离矩阵 [B, k_chunk]，
        然后取各 chunk 中的最小值，最终在 chunk 之间再取全局最小。

        Args:
            R (Tensor): 当前残差，形状 [B, d]
            C (Tensor): 当前层 codebook，形状 [K, d]
            distance_bias (Tensor | None): 可选偏置项，形状 [K]
                                           用于熵正则：让使用频率高的 cluster "变远"

        Returns:
            idx_min (Tensor): 最近 cluster 的索引，形状 [B]
            q (Tensor): 最近 cluster 的向量，形状 [B, d]
        """
        B, d = R.shape
        r_norm = (R * R).sum(1, keepdim=True)   # [B, 1]，||R||² 对每个样本
        idx_min, dist_min = None, None
        K = C.shape[0]

        for s in range(0, K, self.k_chunk):
            e = min(s + self.k_chunk, K)
            Cj = C[s:e]                                         # [Ck, d]
            c_norm = (Cj * Cj).sum(1)                           # [Ck]
            dists = r_norm - 2.0 * (R @ Cj.t()) + c_norm[None, :]  # [B, Ck]

            # 加入距离偏置（熵正则）：使用频率高的 cluster 距离被人为增大
            if distance_bias is not None:
                dists = dists + distance_bias[s:e][None, :]     # [B, Ck]

            d_chunk, i_chunk = torch.min(dists, dim=1)          # [B]
            i_chunk = i_chunk + s                                # 修正为全局 index

            # 跨 chunk 取最小
            if dist_min is None:
                dist_min, idx_min = d_chunk, i_chunk
            else:
                mask = d_chunk < dist_min
                dist_min = torch.where(mask, d_chunk, dist_min)
                idx_min = torch.where(mask, i_chunk, idx_min)

        # 取出最近 cluster 的向量
        q = C.index_select(0, idx_min)                          # [B, d]
        return idx_min, q


# ═══════════════════════════════════════════════════════════
# STEP 3：前向传播（训练 + 推理）
# ═══════════════════════════════════════════════════════════

    def forward(self, z_e: torch.Tensor):
        """
        残差量化前向传播。

        训练期（self.training=True）：
          - 对每层做最近邻查找
          - 用 EMA 更新 codebook（无需 optimizer，梯度不流入 codebook）
          - 计算 commitment loss 和熵正则 loss
          - 返回 STE 量化向量，梯度可以反传给 encoder

        推理期（self.training=False）：
          - 仅做最近邻量化，codebook 不变
          - commitment loss 和 loss_reg 数值上不重要（返回零附近的值）

        Args:
            z_e (Tensor): encoder 输出，形状 [B, d]

        Returns:
            z_q_total (Tensor): STE 量化向量，形状 [B, d]
            ids (List[Tensor]): 各层 token ID，长度 L，每个元素形状 [B]
            commit (Tensor): Commitment Loss（标量）
            loss_reg (Tensor): 熵正则 loss（标量，用于最大化 codebook 使用多样性）
        """
        z_e = z_e.float()
        residual = z_e           # 初始残差 = encoder 输出
        zs = []                  # 各层量化向量的列表
        ids = []                 # 各层 token ID 的列表
        ent_sum = 0.0            # 累积熵（用于计算正则 loss）

        for l in range(self.L):
            C = self.codebooks[l]   # 当前层 codebook [K, d]

            # ---- 计算熵正则的距离偏置 ----
            # 原理：若某 cluster 的使用频率 p_k 过高，则 log(p_k * K) > 0，
            #       给该 cluster 的距离加正偏置，让它"变远"，被选的概率降低。
            # 效果：强迫各 cluster 的使用频率趋于均匀（均匀时熵最大）。
            distance_bias = None
            if self.training and self.entropy_reg > 0:
                with torch.no_grad():
                    # usage = 归一化后的使用频率 [K]
                    usage = self.ema_cluster_size[l] / self.ema_cluster_size[l].sum().clamp_min(1e-9)
                    # 偏置 = entropy_reg * d * log(usage * K)
                    # 当 usage = 1/K（均匀）时，log(usage*K) = 0，偏置为 0
                    # 当 usage > 1/K（过热）时，偏置 > 0，增大该 cluster 的距离
                    distance_bias = self.entropy_reg * self.d * torch.log(
                        (usage * self.K).clamp_min(1e-9)
                    )

            # ---- 最近邻查找 ----
            idx_l, q_l = self._argmin_chunked(residual, C, distance_bias)
            zs.append(q_l)     # [B, d]
            ids.append(idx_l)  # [B]

            # 统计本 batch 各 cluster 被选中次数
            counts = torch.bincount(idx_l, minlength=self.K).to(z_e.dtype)  # [K]

            # ---- EMA 更新 codebook（仅训练期）----
            if self.training:
                with torch.no_grad():
                    decay, eps = self.ema_decay, self.eps

                    # 更新 EMA 使用频率：n_k ← decay * n_k + (1-decay) * count_k
                    self.ema_cluster_size[l].mul_(decay).add_(counts, alpha=1 - decay)

                    # 更新 EMA 向量和：w_k ← decay * w_k + (1-decay) * Σ_{i: assign=k} residual_i
                    dw = torch.zeros_like(self.ema_weight[l])          # [K, d]
                    dw.index_add_(0, idx_l, residual)                  # 累加被分配到该 cluster 的残差
                    self.ema_weight[l].mul_(decay).add_(dw, alpha=1 - decay)

                    # 计算新的 cluster 中心：center_k = w_k / n_k（Laplace 平滑）
                    n = self.ema_cluster_size[l].sum()
                    cluster_size = (
                        (self.ema_cluster_size[l] + eps) / (n + self.K * eps) * n
                    )                                                  # [K]，平滑后的频率
                    new_cb = self.ema_weight[l] / cluster_size.unsqueeze(1)  # [K, d]
                    C.copy_(new_cb)                                    # 原地更新 codebook 权重

            # ---- 残差推进 ----
            # 下一层的输入 = 当前残差 - 当前层的量化向量
            residual = residual - q_l

            # ---- 累积批内熵（用于计算正则 loss）----
            if self.entropy_reg > 0:
                with torch.no_grad():
                    p = counts / counts.sum().clamp_min(1e-9)           # 归一化为概率分布
                    # Shannon 熵：H = -Σ p_k * log(p_k)，均匀时最大（log K）
                    ent = -(p * p.clamp_min(1e-9).log()).sum()
                ent_sum = ent_sum + ent

        # ---- 计算最终输出 ----
        # 重建向量 = 各层量化向量之和（近似还原 z_e）
        z_q_total = sum(zs)                                            # [B, d]

        # STE（直通估计器）：前向用量化值，反向梯度直通给 z_e
        z_q_total = z_e + (z_q_total - z_e).detach()

        # Commitment Loss：鼓励 encoder 输出靠近码本，权重 beta
        commit = self.beta * F.mse_loss(z_e, z_q_total.detach())

        # 熵正则 Loss（取负，因为我们想最大化熵）
        loss_reg = -self.entropy_reg * (ent_sum / max(1, self.L))

        return z_q_total, ids, commit, loss_reg


# ═══════════════════════════════════════════════════════════
# STEP 4：保存 / 加载 codebook
# ═══════════════════════════════════════════════════════════

    def get_codebooks_numpy(self) -> np.ndarray:
        """
        将所有层的 codebook 导出为 numpy 数组。

        Returns:
            np.ndarray: 形状 [L, K, d]，可用于保存或传给 RQ-KMeans.init_from_rqkmeans()
        """
        return np.stack([
            p.detach().cpu().numpy() for p in self.codebooks
        ], axis=0)

    def encode(self, Z: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        """
        推理阶段：批量将 embedding 编码为 token 序列（不更新 codebook）。

        Args:
            Z (Tensor): Item embedding，形状 [N, d]
            batch_size (int): 每批处理的 Item 数量

        Returns:
            np.ndarray: codes，形状 [N, L]，dtype=int32
        """
        self.eval()                        # 切换到推理模式，EMA 不更新
        N = Z.shape[0]
        all_ids = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = Z[i:i + batch_size].float()
                _, ids, _, _ = self.forward(batch)
                # ids: List[L] of [B] → 转置为 [B, L]
                batch_codes = torch.stack(ids, dim=1).cpu().numpy()
                all_ids.append(batch_codes)

        return np.concatenate(all_ids, axis=0).astype(np.int32)


# ═══════════════════════════════════════════════════════════
# 演示：完整训练流程
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("RQ-VAE 演示：Item Embedding → Semantic ID（EMA 在线训练）")
    print("=" * 60)

    # 设备选择：优先 MPS（Apple Silicon），其次 CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[设备] 使用：{device}")

    # 模拟 1000 个 Item，embedding 维度 = 64
    torch.manual_seed(42)
    N, d = 1000, 64
    embeddings = torch.randn(N, d).to(device)

    # ---- 初始化 RQ-VAE ----
    model = ResidualQuantizer(
        L=3,               # 3 层残差量化
        K=32,              # 每层 32 个 cluster（码本大小）
        d=d,               # embedding 维度
        beta=0.25,         # commitment loss 权重
        entropy_reg=1e-4,  # 熵正则权重
        k_chunk=16,        # 分块大小（演示用，实际可设 512~2048）
        ema_decay=0.99,    # EMA 衰减
    ).to(device)

    print(f"\n[1] 模型参数：L={model.L}, K={model.K}, d={model.d}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Codebook 总参数量：{total_params}（{model.L} 层 × {model.K} × {model.d}）")

    # ---- 模拟训练（几个 epoch）----
    print(f"\n[2] 模拟训练（3 个 epoch，batch_size=64）...")
    model.train()
    batch_size = 64
    for epoch in range(3):
        total_commit = 0.0
        total_entropy = 0.0
        n_batch = 0

        for i in range(0, N, batch_size):
            batch = embeddings[i:i + batch_size]
            z_q, ids, commit, loss_reg = model(batch)

            # 总 loss = commitment loss + 熵正则 loss
            # 注意：EMA 直接更新 codebook，无需对 codebook 参数做梯度下降
            # 此处只需对 encoder（如果有）做反传。这里演示只有量化器，没有 encoder，
            # 所以 loss 仅用于观察，不实际调用 backward()。
            total_commit += commit.item()
            total_entropy += (-loss_reg).item()   # 取负是为了显示"熵值"（越大越好）
            n_batch += 1

        print(f"    Epoch {epoch+1}: commit_loss={total_commit/n_batch:.6f}, "
              f"avg_entropy={total_entropy/n_batch:.4f}")

    # ---- 推理：编码全量 Item ----
    print(f"\n[3] 推理：编码全量 {N} 个 Item...")
    model.eval()
    codes = model.encode(embeddings)
    print(f"    codes 形状：{codes.shape}")      # (1000, 3)
    print(f"    前 3 个 Item 的 codes：\n    {codes[:3]}")

    # ---- 统计 codebook 使用率 ----
    print(f"\n[4] Codebook 使用率统计（第 1 层）：")
    layer0_codes = codes[:, 0]
    unique, counts = np.unique(layer0_codes, return_counts=True)
    print(f"    使用了 {len(unique)} / {model.K} 个 cluster")
    print(f"    最多分配：{counts.max()} 个 Item，最少：{counts.min()} 个 Item")
    print(f"    均匀度（越接近 1 越均匀）：{counts.min() / counts.max():.3f}")

    print("\n演示完成！")


# ═══════════════════════════════════════════════════════════
# ASCII 数据流图：RQ-VAE 完整执行过程
# ═══════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  阶段 A：forward(z_e) —— 单次 batch 的训练/推理完整流程           │
# │                                                                  │
# │  z_e [B,d]  ──────────────────────────────────────────────────┐ │
# │  (encoder 输出)                                                │ │
# │                                                                │ │
# │  residual = z_e   ← 初始残差 = encoder 输出本身                │ │
# │                                                                │ │
# │  ╔══════════ Layer 1 ══════════════════════════════════════╗   │ │
# │  ║                                                         ║   │ │
# │  ║  ① 熵正则偏置（training=True 且 entropy_reg>0 时）       ║   │ │
# │  ║    usage[K] = ema_cluster_size[0] / sum                 ║   │ │
# │  ║    bias[K]  = entropy_reg * d * log(usage * K)         ║   │ │
# │  ║    → usage 高的 cluster 得到正偏置，等效"变远"           ║   │ │
# │  ║    → 逼迫 item 分散到低频 cluster，均匀化使用             ║   │ │
# │  ║                                                         ║   │ │
# │  ║  ② 分块最近邻 _argmin_chunked(residual, C, bias)        ║   │ │
# │  ║    r_norm[B,1] = ||residual||²                          ║   │ │
# │  ║    for s in range(0, K, k_chunk):   ← 按 K 分块         ║   │ │
# │  ║      Cj = C[s:s+k_chunk]  [Ck,d]                       ║   │ │
# │  ║      dist[B,Ck] = r_norm - 2*(R@Cj.T) + ||Cj||²        ║   │ │
# │  ║      dist += bias[s:s+k_chunk]  (若有偏置)              ║   │ │
# │  ║      跨 chunk 取全局最小 → idx[B], q[B,d]               ║   │ │
# │  ║                                                         ║   │ │
# │  ║  ③ EMA 更新码本（training=True 时，无需 optimizer）      ║   │ │
# │  ║    counts[K] = bincount(idx)   ← 本 batch 各 cluster    ║   │ │
# │  ║                                   被选中次数             ║   │ │
# │  ║    ema_size[l]   ← 0.99*ema_size   + 0.01*counts        ║   │ │
# │  ║    ema_weight[l] ← 0.99*ema_weight + 0.01*(Σresidual)  ║   │ │
# │  ║                                   ↑ 累加残差（不是z_e）  ║   │ │
# │  ║    new_center = ema_weight / Laplace(ema_size)          ║   │ │
# │  ║    codebook[l].copy_(new_center)  ← 原地更新             ║   │ │
# │  ║                                                         ║   │ │
# │  ║  ④ 残差推进                                              ║   │ │
# │  ║    residual = residual - q   ← 下一层拟合这个剩余误差    ║   │ │
# │  ║                                                         ║   │ │
# │  ║  ⑤ 熵统计（计算正则 loss 用）                            ║   │ │
# │  ║    p[K] = counts / sum(counts)                          ║   │ │
# │  ║    H_l  = -Σ p_k * log(p_k)   ← 本层批内香农熵          ║   │ │
# │  ╚═════════════════════════════════════════════════════════╝   │ │
# │       │                                                        │ │
# │       ▼  (residual 越来越小，L 层后接近零向量)                  │ │
# │  ╔══════════ Layer 2 ══════════════════════════════════════╗   │ │
# │  ║  重复 ①②③④⑤，codebook[1] 独立训练                      ║   │ │
# │  ╚═════════════════════════════════════════════════════════╝   │ │
# │       │                                                        │ │
# │       ▼                                                        │ │
# │  ╔══════════ Layer L ══════════════════════════════════════╗   │ │
# │  ║  重复 ①②③④⑤，codebook[L-1] 独立训练                    ║   │ │
# │  ╚═════════════════════════════════════════════════════════╝   │ │
# │                                                                │ │
# │  ─── 汇总阶段 ─────────────────────────────────────────────   │ │
# │                                                                │ │
# │  z_q_total = q1 + q2 + ... + qL   [B,d]  ← 各层量化向量之和   │ │
# │                                                                │ │
# │  STE（直通估计器）：                                            │ │
# │  ┌───────────────────────────────────────────────────────┐    │ │
# │  │  z_q_ste = z_e + (z_q_total - z_e).detach()          │    │ │
# │  │                                                       │    │ │
# │  │  前向传播：值 = z_q_total（量化后，不可微）             │    │ │
# │  │  反向传播：梯度 = ∂z_e（绕过 argmin，直接流向 encoder） │    │ │
# │  └───────────────────────────────────────────────────────┘    │ │
# │                                                                │ │
# │  commit_loss = β * MSE(z_e, z_q_total.detach())               │ │
# │  → 让 encoder 输出靠近 codebook（codebook 由 EMA 控制，不靠此） │ │
# │                                                                │ │
# │  loss_reg = -entropy_reg * mean(H_1, H_2, ..., H_L)          │ │
# │  → 最大化熵（取负号），防止 codebook collapse                   │ │
# │                                                                │ │
# │  返回 (z_q_ste, [idx_1,...,idx_L], commit_loss, loss_reg)     │ │
# └────────────────────────────────────────────────────────────────┘ │
#                                                                     │
# ┌──────────────────────────────────────────────────────────────────┤
# │  阶段 B：encode(Z) —— 推理时批量编码（eval 模式，EMA 不更新）     │
# │                                                                  │
# │  model.eval()  ← 切换推理模式                                    │
# │                                                                  │
# │  for batch in Z:                                                 │
# │    _, ids, _, _ = model.forward(batch)   ← training=False       │
# │    batch_codes = stack(ids, dim=1)  [B, L]                      │
# │                                                                  │
# │  codes [N, L]  ← 全量 Item 的 Semantic ID                       │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  EMA 更新 vs 梯度更新 原理对比                                    │
# │                                                                  │
# │  梯度更新方式：                                                   │
# │    codebook ∈ nn.Parameter → 参与 optimizer.step()               │
# │    通过 MSE loss 的梯度调整 codebook 位置                         │
# │    问题：codebook 梯度方向和 ema 方向不一致时，训练震荡           │
# │                                                                  │
# │  EMA 更新方式（本实现）：                                         │
# │    codebook 不走 optimizer                                       │
# │    new_center_k = EMA_weight_k / EMA_size_k                     │
# │                 ≈ 被分配到 cluster k 的所有样本的移动平均位置     │
# │    → 等价于"慢速 K-Means 在线更新"                               │
# │    → 训练更稳定，对 lr 不敏感                                     │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  熵正则防 Collapse 机制示意                                       │
# │                                                                  │
# │  无正则时：                                                       │
# │    cluster 3 处于数据密集区 → 被大量选中 → EMA 越来越"中心"       │
# │    cluster 7 处于稀疏区 → 很少被选 → EMA 逐渐偏离数据分布        │
# │    最终：cluster 3 霸占 80% item，codebook 等效从 K 缩为几个     │
# │                                                                  │
# │  加熵正则后：                                                     │
# │    usage[3] 高 → bias[3] = entropy_reg*d*log(usage[3]*K) >> 0   │
# │    → cluster 3 的距离被人为增大                                   │
# │    → 部分 item 改选 cluster 5/7/...                              │
# │    → 使用率趋于均匀，实际有效 cluster 数维持在 K 附近             │
# └──────────────────────────────────────────────────────────────────┘
#
# 类方法速查：
#   ResidualQuantizer(nn.Module)
#     init_from_rqkmeans(codebooks_np) ← 从 RQ-KMeans 热启动
#     _argmin_chunked(R, C, bias)      ← 内部：分块最近邻
#     forward(z_e)                     ← 训练+推理主流程
#     get_codebooks_numpy()            ← 导出 codebook [L,K,d]
#     encode(Z, batch_size)            ← 推理批量编码 → codes [N,L]

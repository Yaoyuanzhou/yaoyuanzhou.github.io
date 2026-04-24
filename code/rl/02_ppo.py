"""
PPO（Proximal Policy Optimization）完整训练循环
================================================
对应文档：topics/notes-rl.html §4

PPO 在 RLHF 中的核心流程：
    每个 batch：
        1. Actor 生成一批回答（推理，不算梯度）
        2. Reward Model 打分 → r_final = r_phi - beta * KL
        3. Critic 估计每个 token 的 V(s_t)
        4. GAE 计算每个 token 的 Advantage A_t
        5. PPO-Clip 更新 Actor（多次 mini-batch）
        6. MSE 更新 Critic

Actor loss（完整展开）：
    L_actor = -E_t[ min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t) ] - c2 * S[pi]
    其中 rho_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

Critic loss：
    L_critic = E_t[ (V(s_t) - R_hat_t)^2 ]
    其中 R_hat_t = gamma^(T-t) * r_final（越靠前的 token label 越小）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, NamedTuple


# ═══════════════════════════════════════════════════════════
# STEP 1：Actor + Critic 模型
# ═══════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Actor 和 Critic 共享 Transformer 主干，只有输出头不同。

    Actor 输出头：  Linear(hidden → vocab_size) → token 概率分布
    Critic 输出头： Linear(hidden → 1)           → 状态价值 V(s_t)

    文档对应：§4.4 四模型中的 ① Actor 和 ④ Critic
    """

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4,
            dim_feedforward=hidden_size * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor 头：输出每个位置的 token 概率分布
        self.actor_head  = nn.Linear(hidden_size, vocab_size)
        # Critic 头：输出每个位置的状态价值 V(s_t)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: (batch, seq_len)

        Returns:
            logits:  (batch, seq_len, vocab_size)  Actor 的 token 分布
            values:  (batch, seq_len)               Critic 的 V(s_t) 估值
        """
        x       = self.embedding(input_ids)   # (batch, seq_len, hidden)
        x       = self.transformer(x)
        logits  = self.actor_head(x)           # (batch, seq_len, vocab_size)
        values  = self.critic_head(x).squeeze(-1)  # (batch, seq_len)
        return logits, values

    def get_log_prob(self, input_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """
        计算 response 部分每个 token 的 log 概率。

        Args:
            input_ids:    (batch, prompt_len + resp_len) 完整序列
            response_ids: (batch, resp_len) 仅 response 部分的 token id

        Returns:
            log_probs: (batch, resp_len)
        """
        logits, _ = self.forward(input_ids)
        # 取 response 对应位置的 logits（prompt 之后）
        resp_len      = response_ids.shape[1]
        resp_logits   = logits[:, -resp_len:, :]          # (batch, resp_len, vocab)
        log_probs_all = F.log_softmax(resp_logits, dim=-1)
        # 取实际生成的 token 对应的 log 概率
        log_probs = log_probs_all.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)  # (batch, resp_len)
        return log_probs


# ═══════════════════════════════════════════════════════════
# STEP 2：GAE 计算优势估计
# ═══════════════════════════════════════════════════════════

def compute_gae(
    rewards:   torch.Tensor,   # (batch, seq_len) 每步即时奖励（只有终态非零）
    values:    torch.Tensor,   # (batch, seq_len) Critic 估值 V(s_t)
    gamma:     float = 0.99,
    lam:       float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE（广义优势估计）。

    核心公式：
        delta_t  = r_t + gamma * V(s_{t+1}) - V(s_t)   ← TD 误差（Bellman 恒等式的偏差）
        A_t      = delta_t + (gamma*lam) * A_{t+1}      ← 从后往前递推

    delta_t > 0：Critic 低估了，这步实际比预期好
    delta_t < 0：Critic 高估了，实际比预期差

    Args:
        rewards: (batch, seq_len) 每步即时奖励，LLM 里只有 r[:, -1] 非零
        values:  (batch, seq_len) Critic 的估值

    Returns:
        advantages:  (batch, seq_len)  每个 token 的优势估计 A_t
        returns:     (batch, seq_len)  目标价值 R_hat_t = A_t + V(s_t)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae        = torch.zeros(batch_size)

    # 从后往前递推：A_T = delta_T，A_t = delta_t + gamma*lam * A_{t+1}
    for t in reversed(range(seq_len)):
        next_value = values[:, t + 1] if t < seq_len - 1 else torch.zeros(batch_size)
        # TD 误差：r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        # GAE 递推
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae

    returns = advantages + values   # R_hat_t = A_t + V(s_t)
    return advantages, returns


# ═══════════════════════════════════════════════════════════
# STEP 3：PPO-Clip Actor loss
# ═══════════════════════════════════════════════════════════

def ppo_clip_loss(
    log_probs_new: torch.Tensor,    # (batch, resp_len) 当前策略的 log 概率
    log_probs_old: torch.Tensor,    # (batch, resp_len) 旧策略的 log 概率（采样时固定）
    advantages:    torch.Tensor,    # (batch, resp_len) GAE 优势估计
    clip_eps:      float = 0.2,
    entropy_coef:  float = 0.01,
    logits:        torch.Tensor = None,  # (batch, resp_len, vocab) 用于计算熵
) -> torch.Tensor:
    """
    PPO-Clip 损失函数（完整版，含熵正则）。

    公式：
        L_actor = -E_t[ min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t) ] - c2 * S[pi]
        rho_t = exp(log_pi_new - log_pi_old) = pi_new / pi_old

    为什么有负号：
        PyTorch 做梯度下降（最小化 loss）。
        要最大化期望回报，需要加负号：min_loss(-expected_return)。
        当 A_t > 0，梯度下降会增大 rho_t（升高概率）；
        当 A_t < 0，梯度下降会减小 rho_t（降低概率）。
        clip 限制每次改变幅度不超过 eps = 0.2（即 ±20%）。
    """
    # 概率比 rho_t = pi_new / pi_old
    ratio = torch.exp(log_probs_new - log_probs_old)

    # 两项取 min，保证对"好动作"不会无限增大，对"坏动作"不会无限压低
    unclipped = ratio * advantages
    clipped   = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # 熵正则：鼓励 Actor 保持一定探索性，防止过早坍塌成确定性策略
    entropy_loss = 0.0
    if logits is not None:
        probs        = F.softmax(logits, dim=-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        entropy      = -(probs * log_probs_all).sum(-1).mean()
        entropy_loss = -entropy_coef * entropy  # 最大化熵 = 最小化负熵

    return policy_loss + entropy_loss


# ═══════════════════════════════════════════════════════════
# STEP 4：KL 惩罚（Reference 模型）
# ═══════════════════════════════════════════════════════════

def compute_kl_penalty(
    log_probs_actor: torch.Tensor,   # (batch, resp_len)
    log_probs_ref:   torch.Tensor,   # (batch, resp_len) 冻结的 SFT 参考模型
    beta:            float = 0.05,
) -> torch.Tensor:
    """
    KL 散度惩罚，防止 Actor 偏离 SFT 参考模型太远。

    公式：KL[pi_theta || pi_ref] ≈ log_pi_theta - log_pi_ref
    最终奖励：r_final = r_phi - beta * KL

    文档对应：§4.5 第一步"构造最终奖励信号"
    """
    kl = log_probs_actor - log_probs_ref   # (batch, resp_len)
    return beta * kl


# ═══════════════════════════════════════════════════════════
# STEP 5：完整 PPO 一个 batch 的训练流程
# ═══════════════════════════════════════════════════════════

def ppo_train_step(
    actor_critic:     ActorCritic,
    ref_model:        ActorCritic,      # 冻结的参考模型（SFT 后冻结）
    prompt_ids:       torch.Tensor,     # (batch, prompt_len)
    response_ids:     torch.Tensor,     # (batch, resp_len) Actor 生成的回答
    reward_scores:    torch.Tensor,     # (batch,) RM 打的分（终态标量）
    optimizer:        torch.optim.Optimizer,
    gamma: float = 0.99,
    lam:   float = 0.95,
    beta:  float = 0.05,
    clip_eps: float = 0.2,
    n_ppo_epochs: int = 4,
) -> dict:
    """
    PPO 一个 batch 的完整流程。

    流程：
        1. 用旧策略的 log_probs 作为基准（采样时已固定）
        2. 构造奖励：r_t = 0（中间步），r_T = reward_score - beta * KL_T
        3. 用 Critic 的 V(s_t) + GAE 算出每个 token 的 A_t
        4. 对这批数据做 n_ppo_epochs 次 PPO 更新
    """
    batch_size, resp_len = response_ids.shape
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (batch, total_len)

    # ── 第一步：计算参考模型 log_probs（不算梯度，因为 ref 是冻结的）──
    with torch.no_grad():
        log_probs_ref = ref_model.get_log_prob(input_ids, response_ids)  # (batch, resp_len)

    # ── 第二步：计算旧策略 log_probs 和 Critic 估值（采样时的基准）──
    with torch.no_grad():
        log_probs_old, values_old = _get_actor_output(actor_critic, input_ids, response_ids)

    # ── 第三步：构造每步奖励 r_t ──
    # 只有终态（最后一个 response token）有奖励，中间步都是 0
    rewards = torch.zeros(batch_size, resp_len)
    kl_last = log_probs_old[:, -1] - log_probs_ref[:, -1]
    rewards[:, -1] = reward_scores - beta * kl_last   # r_final = r_phi - beta * KL

    # ── 第四步：GAE ──
    advantages, returns = compute_gae(rewards, values_old, gamma, lam)
    # 标准化优势（减均值除标准差，稳定训练）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── 第五步：PPO 多轮更新 ──
    total_actor_loss  = 0.0
    total_critic_loss = 0.0

    for _ in range(n_ppo_epochs):
        optimizer.zero_grad()

        # 当前策略的 log_probs 和 Critic 估值
        log_probs_new, values_new = _get_actor_output(actor_critic, input_ids, response_ids)
        logits_new, _ = actor_critic(input_ids)
        resp_logits   = logits_new[:, -resp_len:, :]

        # Actor loss（PPO-Clip + 熵正则）
        actor_loss  = ppo_clip_loss(log_probs_new, log_probs_old, advantages,
                                    clip_eps=clip_eps, logits=resp_logits)
        # Critic loss（MSE 回归到目标价值 R_hat_t）
        critic_loss = F.mse_loss(values_new[:, -resp_len:], returns)

        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=1.0)
        optimizer.step()

        total_actor_loss  += actor_loss.item()
        total_critic_loss += critic_loss.item()

    return {
        "actor_loss":  total_actor_loss  / n_ppo_epochs,
        "critic_loss": total_critic_loss / n_ppo_epochs,
    }


def _get_actor_output(
    model: ActorCritic,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取 Actor 的 log_probs 和 Critic 的 values。"""
    logits, values = model(input_ids)
    resp_len      = response_ids.shape[1]
    resp_logits   = logits[:, -resp_len:, :]
    log_probs_all = F.log_softmax(resp_logits, dim=-1)
    log_probs     = log_probs_all.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return log_probs, values


# ═══════════════════════════════════════════════════════════
# 主程序：演示完整 PPO 训练步骤
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PPO-RLHF 训练循环演示")
    print("=" * 60)

    vocab_size  = 1000
    hidden_size = 128
    prompt_len  = 8
    resp_len    = 16
    batch_size  = 4

    # 1. 初始化模型
    actor_critic = ActorCritic(vocab_size, hidden_size, num_layers=1)
    ref_model    = ActorCritic(vocab_size, hidden_size, num_layers=1)
    ref_model.load_state_dict(actor_critic.state_dict())  # ref = SFT 模型快照
    for p in ref_model.parameters():
        p.requires_grad_(False)  # 完全冻结

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)

    print(f"\nActor/Critic 参数量：{sum(p.numel() for p in actor_critic.parameters()):,}")

    # 2. 模拟训练数据
    prompt_ids   = torch.randint(1, vocab_size, (batch_size, prompt_len))
    response_ids = torch.randint(1, vocab_size, (batch_size, resp_len))
    reward_scores = torch.randn(batch_size)   # 模拟 RM 打分

    print(f"\n模拟数据：batch={batch_size}, prompt_len={prompt_len}, resp_len={resp_len}")
    print(f"RM 打分（r_phi）：{reward_scores.tolist()}")

    # 3. 执行一个 PPO 训练步骤
    print("\n--- PPO 训练步骤 ---")
    losses = ppo_train_step(
        actor_critic, ref_model,
        prompt_ids, response_ids, reward_scores,
        optimizer,
        n_ppo_epochs=4,
    )
    print(f"Actor Loss:  {losses['actor_loss']:.4f}")
    print(f"Critic Loss: {losses['critic_loss']:.4f}")

    print("\n--- PPO 核心流程说明 ---")
    print("r_phi(x,y) 不直接在 loss 里——它先经 GAE 变换，变成每个 token 的 A_t")
    print("A_t > 0：这个 token 比 Critic 预期好 → rho_t 增大 → 概率升高")
    print("A_t < 0：这个 token 比 Critic 预期差 → rho_t 减小 → 概率降低")
    print("clip 限制 rho_t ∈ [0.8, 1.2]，防止单次更新过猛")

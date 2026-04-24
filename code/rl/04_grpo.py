"""
GRPO（Group Relative Policy Optimization）组相对策略优化
=========================================================
对应文档：topics/notes-rl.html §6

GRPO 是 DeepSeek-R1 使用的核心训练算法，由 DeepSeek 团队提出。
相比 PPO，GRPO 的最大创新是：完全去掉了 Critic（价值模型）。

GRPO 的核心思路：
    对同一个 prompt，让 Actor 生成 G 条回答（一组）。
    用这 G 条回答的奖励分数，在组内做归一化，得到每条回答的相对优势。
    A_i = (r_i - mean(r)) / std(r)   ← 组内归一化的 Advantage

GRPO vs PPO：
    PPO  需要 Critic 估计 V(s_t)，并用 GAE 计算每个 token 的 A_t
    GRPO 直接用组内相对排名作为 Advantage，不需要 Critic，不需要 GAE

为什么可以这样做：
    V(s_t) 的本质是"平均基线"。GRPO 用组内均值代替 Critic 提供的基线——
    对同一个 prompt，G 条回答的均值就是一个自然的"平均期望"。

代价：
    每个 prompt 要生成 G 条（通常 G=8~16），推理成本是 PPO 的 G 倍。
    但省掉了 Critic 的训练和存储，总体往往更高效。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable


# ═══════════════════════════════════════════════════════════
# STEP 1：简化版语言模型（与 PPO/DPO 结构相同）
# ═══════════════════════════════════════════════════════════

class PolicyModel(nn.Module):
    """GRPO 的策略模型（Actor），无需 Critic 头。"""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4,
            dim_feedforward=hidden_size * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head     = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.lm_head(x)  # (batch, seq_len, vocab_size)

    def get_log_probs(
        self,
        input_ids:    torch.Tensor,   # (batch, total_len)
        response_ids: torch.Tensor,   # (batch, resp_len)
    ) -> torch.Tensor:
        """计算 response 每个 token 的 log 概率，求和得到整条序列的 log prob。"""
        logits    = self.forward(input_ids)
        resp_len  = response_ids.shape[1]
        resp_logits = logits[:, -resp_len:, :]                # (batch, resp_len, vocab)
        log_probs   = F.log_softmax(resp_logits, dim=-1)
        token_lp    = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_lp.sum(dim=-1)                            # (batch,)


# ═══════════════════════════════════════════════════════════
# STEP 2：组内相对优势计算
# ═══════════════════════════════════════════════════════════

def compute_group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO 的核心：组内归一化得到 Advantage。

    公式：
        A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)

    直觉：
        - 奖励高于组内均值 → A > 0，比同组其他回答好
        - 奖励低于组内均值 → A < 0，比同组其他回答差
        - 不需要 Critic 提供基线，用组内均值作为自然基线

    Args:
        rewards: (n_prompts, G) 每个 prompt 的 G 条回答的奖励分数

    Returns:
        advantages: (n_prompts, G) 归一化后的组内相对优势
    """
    mean_r = rewards.mean(dim=1, keepdim=True)   # (n_prompts, 1)
    std_r  = rewards.std(dim=1, keepdim=True)    # (n_prompts, 1)
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    return advantages


# ═══════════════════════════════════════════════════════════
# STEP 3：GRPO 损失函数
# ═══════════════════════════════════════════════════════════

def grpo_loss(
    model:           PolicyModel,
    ref_model:       PolicyModel,
    prompt_ids:      torch.Tensor,        # (n_prompts, prompt_len)
    response_ids:    torch.Tensor,        # (n_prompts * G, resp_len) 展平后的所有回答
    advantages_flat: torch.Tensor,        # (n_prompts * G,) 展平后的优势
    clip_eps:        float = 0.2,
    beta:            float = 0.01,        # KL 惩罚系数（GRPO 通常较小）
) -> torch.Tensor:
    """
    GRPO 损失（PPO-Clip 形式，无 Critic）。

    L_GRPO = -E[ min(rho * A, clip(rho, 1-eps, 1+eps) * A) ] + beta * KL

    和 PPO 的区别：
        PPO：A_t 是 GAE 算出的每个 token 的优势（需要 Critic）
        GRPO：A_i 是整条 response 的组内归一化优势（不需要 Critic）
    """
    # 拼接 prompt + response（每条 response 对应其 prompt）
    n_total  = response_ids.shape[0]
    G        = n_total // prompt_ids.shape[0]
    # 把 prompt 重复 G 次，和 response 对齐
    prompt_expanded = prompt_ids.repeat_interleave(G, dim=0)  # (n_prompts*G, prompt_len)
    input_ids       = torch.cat([prompt_expanded, response_ids], dim=1)

    # 当前模型的 log 概率
    log_probs_new = model.get_log_probs(input_ids, response_ids)       # (n_total,)

    # 参考模型的 log 概率（冻结）
    with torch.no_grad():
        log_probs_ref = ref_model.get_log_probs(input_ids, response_ids)  # (n_total,)
        log_probs_old = log_probs_new.detach()                             # 采样时固定

    # 概率比 rho = pi_theta / pi_old
    ratio = torch.exp(log_probs_new - log_probs_old)

    # PPO-Clip
    unclipped = ratio * advantages_flat
    clipped   = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages_flat
    policy_loss = -torch.min(unclipped, clipped).mean()

    # KL 惩罚：防止偏离参考模型
    kl_loss = beta * (log_probs_new - log_probs_ref).mean()

    return policy_loss + kl_loss


# ═══════════════════════════════════════════════════════════
# STEP 4：奖励函数（规则/模型皆可）
# ═══════════════════════════════════════════════════════════

def rule_based_reward(responses: List[str]) -> torch.Tensor:
    """
    基于规则的奖励函数（GRPO 常见做法，尤其在数学/代码任务上）。

    真实 DeepSeek-R1 的奖励规则：
        - 答案正确 → +1
        - 答案错误 → 0
        - 格式正确（有 <think> 标签）→ 额外 +0.1

    这里用简单规则模拟：回答越长且包含关键词得分越高。
    """
    rewards = []
    for resp in responses:
        score = 0.0
        # 规则1：包含数字（模拟"包含计算过程"）
        if any(c.isdigit() for c in resp):
            score += 0.5
        # 规则2：长度合理（10~50 个字符）
        if 10 <= len(resp) <= 50:
            score += 0.5
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# STEP 5：GRPO 完整训练步骤
# ═══════════════════════════════════════════════════════════

def grpo_train_step(
    model:          PolicyModel,
    ref_model:      PolicyModel,
    optimizer:      torch.optim.Optimizer,
    prompt_ids:     torch.Tensor,          # (n_prompts, prompt_len)
    reward_fn:      Callable,              # 奖励函数
    G:              int   = 8,             # 每个 prompt 生成多少条回答
    resp_len:       int   = 16,
    vocab_size:     int   = 32000,
    clip_eps:       float = 0.2,
    beta:           float = 0.01,
) -> dict:
    """
    GRPO 一个训练步骤。

    流程：
        1. 对每个 prompt 采样 G 条回答
        2. 用奖励函数对每条打分
        3. 组内归一化得到 Advantage
        4. PPO-Clip 更新 Actor（无 Critic）
    """
    n_prompts = prompt_ids.shape[0]

    # ── 第一步：采样 G 条回答（真实场景用 model.generate()，这里模拟）──
    response_ids = torch.randint(1, vocab_size, (n_prompts * G, resp_len))

    # ── 第二步：计算奖励（模拟：随机分数，真实场景调用 RM 或规则函数）──
    # rewards shape: (n_prompts, G)
    rewards_flat = torch.randn(n_prompts * G)
    rewards      = rewards_flat.view(n_prompts, G)

    # ── 第三步：组内归一化 → Advantage ──
    advantages      = compute_group_advantage(rewards)   # (n_prompts, G)
    advantages_flat = advantages.view(-1)                # (n_prompts*G,)

    # ── 第四步：PPO-Clip 更新 Actor ──
    optimizer.zero_grad()
    loss = grpo_loss(
        model, ref_model,
        prompt_ids, response_ids, advantages_flat,
        clip_eps=clip_eps, beta=beta,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss":        loss.item(),
        "mean_reward": rewards.mean().item(),
        "std_reward":  rewards.std().item(),
    }


# ═══════════════════════════════════════════════════════════
# 主程序：演示 GRPO 训练
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO 组相对策略优化训练演示")
    print("=" * 60)

    vocab_size  = 1000
    hidden      = 128
    prompt_len  = 8
    G           = 4    # 每个 prompt 生成 4 条回答
    n_prompts   = 3

    # 1. 初始化模型（无 Critic！）
    model     = PolicyModel(vocab_size, hidden, num_layers=1)
    ref_model = PolicyModel(vocab_size, hidden, num_layers=1)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"\n策略模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"Critic 参数量：0（GRPO 不需要 Critic）")

    # 2. 模拟 prompt
    prompt_ids = torch.randint(1, vocab_size, (n_prompts, prompt_len))

    # 3. 演示组内归一化
    print(f"\n--- 组内 Advantage 计算演示（G={G} 条回答/prompt）---")
    fake_rewards = torch.tensor([
        [0.2, 0.8, 0.5, 1.0],   # prompt 0 的 4 条回答奖励
        [0.1, 0.1, 0.9, 0.3],   # prompt 1
        [0.6, 0.6, 0.6, 0.6],   # prompt 2（奖励完全相同时 std=0）
    ])
    advantages = compute_group_advantage(fake_rewards)
    for i in range(n_prompts):
        print(f"  Prompt {i}: 奖励={fake_rewards[i].tolist()} → A={advantages[i].round(decimals=2).tolist()}")

    # 4. 训练一步
    print("\n--- GRPO 训练步骤 ---")
    metrics = grpo_train_step(
        model, ref_model, optimizer,
        prompt_ids, reward_fn=rule_based_reward,
        G=G, resp_len=16, vocab_size=vocab_size,
    )
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")

    print("\n--- GRPO vs PPO 核心区别 ---")
    print("PPO：4 个模型（Actor + Reference + RM + Critic），需要 GAE")
    print("GRPO：2 个模型（Actor + Reference），用组内均值作为基线，无需 Critic/GAE")
    print(f"代价：每个 prompt 要生成 G={G} 条回答，推理成本 = PPO 的 {G} 倍")

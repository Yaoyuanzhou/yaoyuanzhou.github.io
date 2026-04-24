"""
DPO（Direct Preference Optimization）直接偏好优化
==================================================
对应文档：topics/notes-rl.html §5

DPO 的核心思路：
    不需要单独的奖励模型，不需要 PPO 的 RL 循环。
    直接用人类偏好对 (chosen, rejected) 来更新语言模型参数。

DPO 的关键洞察：
    RLHF 最终优化的语言模型和参考模型之间满足：
        r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)) + Z(x)
    把这个关系代入 Bradley-Terry 偏好模型，消掉 r(x,y)，得到 DPO loss：

        L_DPO = -E[ log sigma( beta * (
            log pi_theta(y_w|x) - log pi_ref(y_w|x) -
            log pi_theta(y_l|x) + log pi_ref(y_l|x)
        ))]

    其中 y_w = chosen（胜者），y_l = rejected（败者）

DPO vs PPO：
    PPO：需要 4 个模型（Actor + Reference + RM + Critic），工程复杂
    DPO：只需要 2 个模型（当前模型 + 参考模型），无需 RL 循环，简单高效
    代价：DPO 优化的是"隐式奖励"，不如 PPO 灵活，对数据质量要求更高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════
# STEP 1：语言模型（Actor 和 Reference 共用结构）
# ═══════════════════════════════════════════════════════════

class LanguageModel(nn.Module):
    """
    简化版语言模型，用于演示 DPO。

    DPO 需要两个模型：
        pi_theta（当前模型）：训练中持续更新
        pi_ref  （参考模型）：从 SFT 模型复制，完全冻结
    """

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
        """
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.lm_head(x)

    def sequence_log_prob(
        self,
        input_ids:    torch.Tensor,  # (batch, prompt_len + resp_len)
        response_ids: torch.Tensor,  # (batch, resp_len)
    ) -> torch.Tensor:
        """
        计算整条 response 的 log 概率之和（token-level sum）。

        这是 DPO 的核心计算：
            log pi(y|x) = sum_t log pi(y_t | x, y_{<t})

        Args:
            input_ids:    完整序列（prompt + response）
            response_ids: response 部分的 token id

        Returns:
            log_probs: (batch,) 整条 response 的 log 概率之和
        """
        logits    = self.forward(input_ids)               # (batch, total_len, vocab)
        resp_len  = response_ids.shape[1]
        # 取 response 对应的 logits
        resp_logits   = logits[:, -resp_len:, :]          # (batch, resp_len, vocab)
        log_probs_all = F.log_softmax(resp_logits, dim=-1)
        # 取每个位置实际 token 的 log 概率
        token_log_probs = log_probs_all.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)  # (batch, resp_len)
        # 整条 response 的 log 概率：sum over tokens
        return token_log_probs.sum(dim=-1)  # (batch,)


# ═══════════════════════════════════════════════════════════
# STEP 2：DPO 损失函数
# ═══════════════════════════════════════════════════════════

def dpo_loss(
    model:        LanguageModel,
    ref_model:    LanguageModel,
    prompt_ids:   torch.Tensor,    # (batch, prompt_len)
    chosen_ids:   torch.Tensor,    # (batch, resp_len)
    rejected_ids: torch.Tensor,    # (batch, resp_len)
    beta:         float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    DPO 损失函数。

    核心公式（来自文档 §5.2）：
        L_DPO = -E[ log σ( beta * (
            log π_θ(y_w|x) - log π_ref(y_w|x) -
            log π_θ(y_l|x) + log π_ref(y_l|x)
        ))]

    直觉：
        括号里的量 = "模型相对参考模型偏好 chosen 的程度" - "模型相对参考模型偏好 rejected 的程度"
        这个量越大 → chosen 越受偏好 → loss 越小（越好）

    Args:
        beta: 控制 KL 偏离惩罚强度，越大越保守（不敢偏离参考模型太多）

    Returns:
        loss: DPO 损失（标量）
        metrics: 包含 chosen/rejected 隐式奖励的字典（用于监控训练）
    """
    # 拼接 prompt + response
    chosen_input   = torch.cat([prompt_ids, chosen_ids],   dim=1)
    rejected_input = torch.cat([prompt_ids, rejected_ids], dim=1)

    # 当前模型的 log 概率
    log_pi_chosen   = model.sequence_log_prob(chosen_input,   chosen_ids)
    log_pi_rejected = model.sequence_log_prob(rejected_input, rejected_ids)

    # 参考模型的 log 概率（不算梯度，ref 是冻结的）
    with torch.no_grad():
        log_ref_chosen   = ref_model.sequence_log_prob(chosen_input,   chosen_ids)
        log_ref_rejected = ref_model.sequence_log_prob(rejected_input, rejected_ids)

    # 隐式奖励：r(x,y) = beta * (log pi_theta - log pi_ref)
    # 文档说明：DPO 把奖励函数"内化"到模型概率的差值里
    implicit_reward_chosen   = beta * (log_pi_chosen   - log_ref_chosen)
    implicit_reward_rejected = beta * (log_pi_rejected - log_ref_rejected)

    # DPO loss = -log σ(r_chosen - r_rejected)
    # 这和 Bradley-Terry 损失形式完全相同，只是"奖励"变成了隐式奖励
    loss = -F.logsigmoid(implicit_reward_chosen - implicit_reward_rejected).mean()

    # 监控指标
    metrics = {
        "implicit_reward_chosen":   implicit_reward_chosen.mean().item(),
        "implicit_reward_rejected": implicit_reward_rejected.mean().item(),
        "reward_margin":            (implicit_reward_chosen - implicit_reward_rejected).mean().item(),
        "accuracy":                 (implicit_reward_chosen > implicit_reward_rejected).float().mean().item(),
    }

    return loss, metrics


# ═══════════════════════════════════════════════════════════
# STEP 3：偏好数据集
# ═══════════════════════════════════════════════════════════

class DPODataset(Dataset):
    """
    DPO 训练数据集。

    每条样本：(prompt, chosen_response, rejected_response)
    来源：和训练 RM 时一样的人类偏好标注数据
    DPO 直接用这些数据，不需要额外训练 RM。
    """

    def __init__(self, size: int = 200, prompt_len: int = 8,
                 resp_len: int = 16, vocab_size: int = 32000):
        self.size       = size
        self.prompt_len = prompt_len
        self.resp_len   = resp_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prompt   = torch.randint(1, self.vocab_size, (self.prompt_len,))
        chosen   = torch.randint(1, self.vocab_size, (self.resp_len,))
        rejected = torch.randint(1, self.vocab_size, (self.resp_len,))
        return prompt, chosen, rejected


# ═══════════════════════════════════════════════════════════
# STEP 4：训练循环
# ═══════════════════════════════════════════════════════════

def train_dpo(
    model:      LanguageModel,
    ref_model:  LanguageModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    lr:         float = 1e-4,
    beta:       float = 0.1,
) -> None:
    """
    DPO 训练循环。

    关键监控指标：
        accuracy：chosen 的隐式奖励 > rejected 的隐式奖励的比例
        reward_margin：两者隐式奖励的差值（越大越好）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss   = 0.0
        total_acc    = 0.0
        total_margin = 0.0
        n_batches    = 0

        for prompt_ids, chosen_ids, rejected_ids in dataloader:
            optimizer.zero_grad()

            loss, metrics = dpo_loss(model, ref_model, prompt_ids, chosen_ids, rejected_ids, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss   += loss.item()
            total_acc    += metrics["accuracy"]
            total_margin += metrics["reward_margin"]
            n_batches    += 1

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {total_loss/n_batches:.4f} | "
              f"Acc: {total_acc/n_batches:.4f} | "
              f"Margin: {total_margin/n_batches:.4f}")


# ═══════════════════════════════════════════════════════════
# 主程序：演示 DPO 训练
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DPO 直接偏好优化训练演示")
    print("=" * 60)

    vocab_size = 1000
    hidden    = 128
    prompt_len = 8
    resp_len   = 12

    # 1. 初始化模型
    model     = LanguageModel(vocab_size, hidden, num_layers=1)
    ref_model = LanguageModel(vocab_size, hidden, num_layers=1)
    ref_model.load_state_dict(model.state_dict())   # ref = SFT 快照
    for p in ref_model.parameters():
        p.requires_grad_(False)   # 完全冻结

    print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"参考模型：冻结，{sum(p.numel() for p in ref_model.parameters()):,} 参数")

    # 2. 数据集
    dataset    = DPODataset(size=200, prompt_len=prompt_len, resp_len=resp_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 训练
    print("\n--- 开始 DPO 训练 ---")
    train_dpo(model, ref_model, dataloader, num_epochs=3, beta=0.1)

    # 4. 验证：对比一对 (chosen, rejected) 的隐式奖励
    print("\n--- 验证：隐式奖励对比 ---")
    prompt   = torch.randint(1, vocab_size, (1, prompt_len))
    chosen   = torch.randint(1, vocab_size, (1, resp_len))
    rejected = torch.randint(1, vocab_size, (1, resp_len))

    with torch.no_grad():
        r_chosen   = 0.1 * (model.sequence_log_prob(torch.cat([prompt, chosen], 1), chosen) -
                            ref_model.sequence_log_prob(torch.cat([prompt, chosen], 1), chosen))
        r_rejected = 0.1 * (model.sequence_log_prob(torch.cat([prompt, rejected], 1), rejected) -
                            ref_model.sequence_log_prob(torch.cat([prompt, rejected], 1), rejected))

    print(f"chosen 隐式奖励：  {r_chosen.item():.4f}")
    print(f"rejected 隐式奖励：{r_rejected.item():.4f}")
    print(f"差值（期望为正）：  {(r_chosen - r_rejected).item():.4f}")

    print("\n--- DPO vs PPO 关键区别 ---")
    print("PPO：需要 RM + Critic + Actor + Reference = 4 个模型，工程复杂")
    print("DPO：只需要 Actor + Reference = 2 个模型，无 RL 循环，简单高效")
    print("代价：DPO 优化隐式奖励，不如 PPO 灵活，对数据质量更敏感")

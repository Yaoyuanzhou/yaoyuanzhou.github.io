"""
RLHF 第二阶段：奖励模型（Reward Model）训练
============================================
对应文档：topics/notes-rl.html §3.2

RLHF 三阶段：
  阶段一：SFT（监督微调）          → 让模型学会"正确格式"的回答
  阶段二：训练奖励模型（本文件）    → 让模型学会"打分"
  阶段三：PPO 强化学习             → 用奖励模型驱动策略模型优化

奖励模型做什么：
  输入：(prompt, response) 对
  输出：一个标量分数（越高越好）
  训练数据：人类标注的偏好对 (prompt, chosen, rejected)
  训练目标：让 chosen 的得分始终高于 rejected（Bradley-Terry 模型）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


# ═══════════════════════════════════════════════════════════
# STEP 1：奖励模型结构
# ═══════════════════════════════════════════════════════════

class RewardModel(nn.Module):
    """
    奖励模型：在语言模型主干上加一个线性输出头。

    结构：
        Transformer 主干（从 SFT 模型复制权重）
        → Linear(hidden_size → 1)
        → 标量分数 r(x, y)

    和 Critic 的区别：
        奖励模型：在训练 RM 阶段训练，训练好后冻结，对整条回答输出一个分数
        Critic：在 PPO 阶段和 Actor 一起训练，对每个 token 输出一个状态价值 V(s_t)
    """

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        # 简化版：用小型 Transformer 模拟真实语言模型主干
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出头：Linear(hidden_size → 1)，把最后一个 token 的隐状态映射成标量
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token id 序列，包含 prompt + response

        Returns:
            reward: (batch,) 每条序列对应的标量奖励分数
        """
        # (batch, seq_len, hidden_size)
        x = self.embedding(input_ids)
        x = self.transformer(x)
        # 取最后一个 token 的隐状态（类似 GPT 的做法）
        last_hidden = x[:, -1, :]                     # (batch, hidden_size)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


# ═══════════════════════════════════════════════════════════
# STEP 2：偏好数据集
# ═══════════════════════════════════════════════════════════

class PreferenceDataset(Dataset):
    """
    Bradley-Terry 偏好数据集。

    每条样本是一个三元组：(prompt, chosen_response, rejected_response)
    人类标注者认为 chosen 比 rejected 更好。

    数据格式（真实训练时）：
        {
            "prompt":    "帮我推荐一本好书",
            "chosen":    "我推荐《百年孤独》……",   ← 人类偏好的回答
            "rejected":  "随便看本书就行了。",       ← 人类不喜欢的回答
        }
    """

    def __init__(self, data: List[Dict], seq_len: int = 32, vocab_size: int = 32000):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 真实场景应用 tokenizer；这里用随机 token 模拟
        chosen_ids   = torch.randint(1, self.vocab_size, (self.seq_len,))
        rejected_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        return chosen_ids, rejected_ids


# ═══════════════════════════════════════════════════════════
# STEP 3：Bradley-Terry 损失函数
# ═══════════════════════════════════════════════════════════

def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry 偏好模型损失。

    核心公式：
        L = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]

    直觉：
        让 chosen 的分数始终高于 rejected 的分数。
        σ(r_chosen - r_rejected) 表示"chosen 比 rejected 好"的概率，
        我们最大化这个概率，即最小化其负对数。

    Args:
        reward_chosen:   (batch,) chosen 回答的奖励分数
        reward_rejected: (batch,) rejected 回答的奖励分数

    Returns:
        loss: 标量损失值
    """
    # log σ(r_c - r_r) = -softplus(r_r - r_c) = -log(1 + exp(r_r - r_c))
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss


# ═══════════════════════════════════════════════════════════
# STEP 4：训练循环
# ═══════════════════════════════════════════════════════════

def train_reward_model(
    model: RewardModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    """
    奖励模型完整训练循环。

    关键监控指标：
        accuracy：在当前 batch 里，chosen 分数 > rejected 分数的比例
        理想情况下应该接近 1.0（100% 正确排序）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        for chosen_ids, rejected_ids in dataloader:
            optimizer.zero_grad()

            r_chosen   = model(chosen_ids)    # (batch,)
            r_rejected = model(rejected_ids)  # (batch,)

            loss = bradley_terry_loss(r_chosen, r_rejected)
            loss.backward()
            optimizer.step()

            # 准确率：chosen 得分 > rejected 得分的比例
            acc = (r_chosen > r_rejected).float().mean().item()

            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_acc  = total_acc  / n_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")


# ═══════════════════════════════════════════════════════════
# STEP 5：推理——对一条 (prompt, response) 打分
# ═══════════════════════════════════════════════════════════

def score_response(model: RewardModel, input_ids: torch.Tensor) -> float:
    """
    用训练好的 RM 对一条回答打分。

    Args:
        model:     训练好的奖励模型
        input_ids: (seq_len,) prompt + response 的 token id

    Returns:
        score: 标量分数（越高越好）
    """
    model.eval()
    with torch.no_grad():
        reward = model(input_ids.unsqueeze(0))  # 加 batch 维度
    return reward.item()


# ═══════════════════════════════════════════════════════════
# 主程序：完整演示
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RLHF 阶段二：奖励模型训练演示")
    print("=" * 60)

    # 1. 构造假数据（真实场景替换为真实偏好标注数据集）
    fake_data = [{"prompt": f"prompt_{i}", "chosen": f"good_{i}", "rejected": f"bad_{i}"}
                 for i in range(200)]
    dataset    = PreferenceDataset(fake_data, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. 初始化奖励模型
    rm = RewardModel(vocab_size=32000, hidden_size=128, num_layers=2)
    print(f"\n奖励模型参数量：{sum(p.numel() for p in rm.parameters()):,}")

    # 3. 训练
    print("\n--- 开始训练 ---")
    train_reward_model(rm, dataloader, num_epochs=3)

    # 4. 推理演示：对比两条不同质量的回答
    print("\n--- 推理演示：对比两条回答的得分 ---")
    good_response = torch.randint(1, 32000, (32,))  # 模拟好回答
    bad_response  = torch.randint(1, 32000, (32,))  # 模拟差回答

    score_good = score_response(rm, good_response)
    score_bad  = score_response(rm, bad_response)
    print(f"好回答得分：{score_good:.4f}")
    print(f"差回答得分：{score_bad:.4f}")
    print(f"差值（期望为正）：{score_good - score_bad:.4f}")

    print("\n--- RM 训练完成，下一步：保存模型，用于 PPO 阶段（见 02_ppo.py）---")

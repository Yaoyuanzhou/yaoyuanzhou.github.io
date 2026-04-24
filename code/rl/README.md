# LLM 强化学习方法代码示例

对应学习笔记：[topics/notes-rl.html](../../topics/notes-rl.html)

---

## 文件列表

| 文件 | 方法 | 对应章节 | 核心内容 |
|------|------|---------|---------|
| `01_reward_model.py` | 奖励模型（RM） | §3 RLHF | Bradley-Terry 损失，偏好对训练 |
| `02_ppo.py` | PPO | §4 PPO | GAE + PPO-Clip + Critic MSE 完整循环 |
| `03_dpo.py` | DPO | §5 DPO | 隐式奖励，直接偏好优化，无需 RM |
| `04_grpo.py` | GRPO | §6 GRPO | 组内归一化 Advantage，无需 Critic |
| `05_rlaif.py` | RLAIF | §7 RLAIF | Judge LLM 自动标注，position bias 缓解 |

---

## 方法对比

```
RLHF（三阶段）
  阶段1：SFT 监督微调
  阶段2：训练 RM（见 01_reward_model.py）← Bradley-Terry 偏好损失
  阶段3：PPO 强化学习（见 02_ppo.py）   ← 4 个模型同时运行

DPO（见 03_dpo.py）
  直接用偏好数据更新模型，无需单独 RM，无需 RL 循环
  只需 2 个模型（Actor + Reference）

GRPO（见 04_grpo.py）
  去掉 Critic，用组内对比代替 GAE
  每个 prompt 生成 G 条回答，组内归一化得到 Advantage

RLAIF（见 05_rlaif.py）
  用 Judge LLM 代替人类标注偏好数据
  两种模式：蒸馏式（先标注再训 RM）/ 直接式（实时调用 Judge）
```

---

## 模型数量对比

| 方法 | 训练时模型数 | 是否需要 Critic | 是否需要独立 RM |
|------|------------|----------------|----------------|
| PPO  | 4（Actor + Reference + RM + Critic） | ✅ | ✅ |
| DPO  | 2（Actor + Reference） | ❌ | ❌ |
| GRPO | 2（Actor + Reference） | ❌ | ✅（或规则奖励） |
| RLAIF | 同 RLHF，但 RM 数据由 AI 生成 | ✅ | ✅（AI 标注） |

---

## 快速运行

```bash
cd code/rl

# 安装依赖
pip install torch

# 运行各方法示例
python 01_reward_model.py   # 奖励模型训练
python 02_ppo.py            # PPO 完整循环
python 03_dpo.py            # DPO 训练
python 04_grpo.py           # GRPO 训练
python 05_rlaif.py          # RLAIF 标注演示
```

---

## 核心公式速查

**Actor loss（PPO）**
$$\mathcal{L} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1\pm\varepsilon)\hat{A}_t\right)\right] - c_2 S[\pi_\theta]$$

**GAE 优势估计**
$$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**DPO loss**
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right)\right]$$

**GRPO 组内 Advantage**
$$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G)}$$

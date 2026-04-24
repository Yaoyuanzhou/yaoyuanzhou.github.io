"""
RLAIF（Reinforcement Learning from AI Feedback）用 AI 代替人类打分
=================================================================
对应文档：topics/notes-rl.html §7

RLAIF 的核心思路：
    RLHF 需要大量人工标注偏好数据（成本高、速度慢）。
    RLAIF 用一个"评判大模型"（Judge LLM，通常是 Claude/GPT-4）代替人类裁判，
    自动生成偏好标注，大幅降低标注成本。

RLAIF 的两种模式（文档 §7.2）：
    模式一："蒸馏式 RLAIF"
        Judge LLM → 生成偏好标注 → 训练 RM → 用 RM 跑 PPO（和 RLHF 相同）
    模式二："直接 RLAIF"
        每次需要奖励时直接调用 Judge LLM → 省掉独立 RM 训练

RLAIF vs RLHF：
    RLHF：人类标注偏好，成本高，质量高但有主观性
    RLAIF：AI 标注偏好，成本低，可大规模扩展，一致性好，但受 Judge LLM 能力限制

关键问题：
    Judge LLM 的偏见会传递给被训练的模型（position bias、verbosity bias 等）
    需要通过 prompt 工程和多轮审查来缓解偏见
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import random


# ═══════════════════════════════════════════════════════════
# STEP 1：Judge LLM 接口（模拟）
# ═══════════════════════════════════════════════════════════

class JudgeLLM:
    """
    评判大模型（Judge LLM）接口。

    真实场景：调用 Claude API 或 GPT-4 API。
    这里模拟 Judge LLM 的行为，用简单规则替代真实 LLM 调用。

    文档说明（§7.1）：
        Judge LLM 收到 prompt：
            "下面两个回答，哪个更好？请只回答 A 或 B。
             问题：{question}
             回答 A：{response_a}
             回答 B：{response_b}"

        返回 "A" 或 "B"，有时加上置信度分数。
    """

    def __init__(self, model_name: str = "claude-3-opus (simulated)"):
        self.model_name = model_name
        print(f"Judge LLM 初始化：{model_name}")

    def compare(
        self,
        question:   str,
        response_a: str,
        response_b: str,
    ) -> Tuple[str, float]:
        """
        让 Judge LLM 比较两个回答。

        Args:
            question:   原始问题
            response_a: 回答 A
            response_b: 回答 B

        Returns:
            winner:     "A" 或 "B"（Judge 认为哪个更好）
            confidence: 置信度 [0, 1]

        真实实现（伪代码）：
            prompt = f'''你是一个公正的评判者。以下哪个回答更好？
            问题：{question}
            回答A：{response_a}
            回答B：{response_b}
            请只回答"A"或"B"，并给出置信度（0-1）。'''
            result = claude_api.complete(prompt)
            winner, confidence = parse_result(result)
        """
        # ── 模拟 Judge LLM 的判断（用简单启发式规则代替真实 LLM）──
        # 规则：更长且包含数字的回答得分更高（模拟"详细度"偏好）
        score_a = len(response_a) * 0.1 + sum(c.isdigit() for c in response_a) * 0.5
        score_b = len(response_b) * 0.1 + sum(c.isdigit() for c in response_b) * 0.5

        # 加入随机噪声（真实 Judge 也会有不确定性）
        score_a += random.gauss(0, 0.3)
        score_b += random.gauss(0, 0.3)

        if score_a > score_b:
            confidence = min(0.95, 0.5 + abs(score_a - score_b) * 0.1)
            return "A", confidence
        else:
            confidence = min(0.95, 0.5 + abs(score_b - score_a) * 0.1)
            return "B", confidence

    def score(self, question: str, response: str) -> float:
        """
        直接给单条回答打分（0~1）。
        用于"直接 RLAIF"模式，省掉比较步骤。
        """
        # 模拟评分
        base_score = len(response) / 100.0
        base_score += sum(c.isdigit() for c in response) * 0.05
        base_score = min(1.0, max(0.0, base_score + random.gauss(0, 0.1)))
        return base_score


# ═══════════════════════════════════════════════════════════
# STEP 2：自动偏好标注（RLAIF 核心）
# ═══════════════════════════════════════════════════════════

class RLAIFAnnotator:
    """
    RLAIF 自动标注器：用 Judge LLM 生成偏好数据。

    文档说明（§7.2）：
        步骤1：给同一个 prompt，让当前策略模型生成多条回答
        步骤2：让 Judge LLM 两两比较，确定偏好顺序
        步骤3：用这些 AI 生成的偏好对训练 RM（和 RLHF 第二阶段相同）

    缓解偏见的技巧（文档 §7.3）：
        - 交换 A/B 位置，取两次结果的平均（消除 position bias）
        - 要求 Judge 先写理由再给答案（chain-of-thought，减少 verbosity bias）
        - 对置信度低的对子降低权重
    """

    def __init__(self, judge: JudgeLLM, confidence_threshold: float = 0.6):
        self.judge = judge
        self.confidence_threshold = confidence_threshold

    def annotate_pair(
        self,
        question:   str,
        response_a: str,
        response_b: str,
    ) -> Optional[Dict]:
        """
        对一对回答进行 AI 偏好标注，并消除 position bias。

        技巧：交换 A/B 位置，做两次判断：
            - 两次一致 → 高置信度，直接使用
            - 两次不一致 → 低置信度，丢弃或降权
        """
        # 第一次判断：A=response_a, B=response_b
        winner1, conf1 = self.judge.compare(question, response_a, response_b)

        # 第二次判断：交换 A/B（消除 position bias）
        winner2, conf2 = self.judge.compare(question, response_b, response_a)
        # 注意：winner2 的 "A" 实际对应 response_b
        winner2_corrected = "B" if winner2 == "A" else "A"

        if winner1 == winner2_corrected:
            # 两次一致，置信度取平均
            avg_confidence = (conf1 + conf2) / 2
            if avg_confidence >= self.confidence_threshold:
                chosen   = response_a if winner1 == "A" else response_b
                rejected = response_b if winner1 == "A" else response_a
                return {
                    "question":    question,
                    "chosen":      chosen,
                    "rejected":    rejected,
                    "confidence":  avg_confidence,
                    "source":      "rlaif",
                }
        # 两次不一致或置信度太低，丢弃
        return None

    def annotate_batch(
        self,
        questions:  List[str],
        responses:  List[List[str]],  # 每个 question 对应多条 response
    ) -> List[Dict]:
        """
        批量标注偏好数据。

        Args:
            questions: 问题列表
            responses: 每个问题对应的多条候选回答（取前两条做比较）

        Returns:
            偏好数据集（格式和 RLHF 的人工标注相同）
        """
        dataset = []
        for q, resps in zip(questions, responses):
            if len(resps) < 2:
                continue
            # 取前两条做比较（真实场景可做更多对比）
            result = self.annotate_pair(q, resps[0], resps[1])
            if result is not None:
                dataset.append(result)
        print(f"标注完成：{len(dataset)}/{len(questions)} 对通过置信度过滤")
        return dataset


# ═══════════════════════════════════════════════════════════
# STEP 3：直接 RLAIF 模式（省掉独立 RM 训练）
# ═══════════════════════════════════════════════════════════

class DirectRLAIFReward:
    """
    直接 RLAIF：每次训练时直接调用 Judge LLM 打分，省掉单独的 RM。

    优点：省掉 RM 训练步骤，流程更简单
    缺点：每次更新都要调用 Judge API，成本更高，速度更慢

    适用场景：小批量、高质量场景，或 Judge LLM API 调用成本可接受时。
    """

    def __init__(self, judge: JudgeLLM, cache: bool = True):
        self.judge = judge
        self.cache: Dict[str, float] = {} if cache else None

    def get_reward(self, question: str, response: str) -> float:
        """
        直接用 Judge LLM 对 (question, response) 打分。
        缓存相同输入的结果，避免重复调用 API。
        """
        cache_key = f"{question}|||{response}"
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]

        score = self.judge.score(question, response)

        if self.cache is not None:
            self.cache[cache_key] = score
        return score

    def get_batch_rewards(
        self,
        questions: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """批量获取奖励分数，返回 tensor 供 PPO/GRPO 使用。"""
        rewards = [self.get_reward(q, r) for q, r in zip(questions, responses)]
        return torch.tensor(rewards, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# STEP 4：RLAIF 完整训练流程（蒸馏式）
# ═══════════════════════════════════════════════════════════

def rlaif_distillation_pipeline(
    judge:       JudgeLLM,
    questions:   List[str],
    responses:   List[List[str]],
) -> List[Dict]:
    """
    蒸馏式 RLAIF 流程：
        1. 用 Judge LLM 自动标注偏好数据
        2. 用标注数据训练 RM（调用 01_reward_model.py 的方式）
        3. 用 RM + PPO 训练 Actor（调用 02_ppo.py 的方式）

    这里只演示第 1 步，后续步骤复用 RLHF 管道。
    """
    print("\n=== 蒸馏式 RLAIF：自动生成偏好数据 ===")
    annotator = RLAIFAnnotator(judge, confidence_threshold=0.55)
    dataset   = annotator.annotate_batch(questions, responses)

    print(f"\n生成的偏好数据样例：")
    for item in dataset[:2]:
        print(f"  问题：{item['question']}")
        print(f"  chosen（AI 偏好）：  {item['chosen'][:40]}...")
        print(f"  rejected（AI 不喜欢）：{item['rejected'][:40]}...")
        print(f"  置信度：{item['confidence']:.2f}")
        print()

    print("下一步：把 dataset 传给 01_reward_model.py 训练 RM，再用 02_ppo.py 跑 PPO")
    return dataset


# ═══════════════════════════════════════════════════════════
# 主程序：演示 RLAIF 两种模式
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RLAIF：用 AI 代替人类打分演示")
    print("=" * 60)

    # 初始化 Judge LLM（模拟）
    judge = JudgeLLM(model_name="claude-3-opus (simulated)")

    # ── 模式一：蒸馏式 RLAIF ──
    questions = [
        "请解释什么是量子纠缠",
        "如何提高代码可读性",
        "推荐一本好书",
    ]
    responses = [
        [
            "量子纠缠是两个粒子之间的神秘联系，测量一个会影响另一个。",
            "量子纠缠是指两个或多个粒子在量子力学框架下形成纠缠态，对其中1个粒子的测量会瞬时影响另一个粒子的状态，无论相距多远。",
        ],
        [
            "写好注释，用有意义的变量名。",
            "提高代码可读性的方法：1) 使用描述性变量名 2) 每个函数只做一件事 3) 添加必要注释 4) 保持一致的代码风格 5) 适当拆分长函数。",
        ],
        [
            "随便看本书吧。",
            "推荐《人类简史》，作者赫拉利用宏观视角讲述人类文明的发展历程，既有学术深度，又通俗易懂，非常适合拓展思维边界。",
        ],
    ]

    dataset = rlaif_distillation_pipeline(judge, questions, responses)

    # ── 模式二：直接 RLAIF ──
    print("\n=== 直接 RLAIF：实时调用 Judge 打分 ===")
    direct_rm = DirectRLAIFReward(judge, cache=True)

    test_questions = ["解释量子纠缠", "推荐好书"]
    test_responses = [
        "量子纠缠是两粒子间的量子关联，观测一个会瞬间影响另一个的状态。",
        "推荐《百年孤独》，马尔克斯的魔幻现实主义杰作。",
    ]

    rewards = direct_rm.get_batch_rewards(test_questions, test_responses)
    for q, r, s in zip(test_questions, test_responses, rewards.tolist()):
        print(f"  问题：{q}")
        print(f"  回答：{r[:40]}...")
        print(f"  Judge 评分：{s:.4f}")
        print()

    print("--- RLAIF vs RLHF 关键区别 ---")
    print("RLHF：人工标注偏好，成本高，质量取决于标注者")
    print("RLAIF：Judge LLM 标注，成本低，一致性好，但有 Judge 的偏见（position bias 等）")
    print("缓解偏见：交换 A/B 位置做两次判断 + CoT 推理 + 置信度过滤")

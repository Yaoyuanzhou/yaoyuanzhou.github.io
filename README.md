# 📚 Paper Notes

个人论文笔记与技术学习站点，使用 GitHub Pages 托管。

🌐 在线访问：**[yaoyuanzhou.github.io](https://yaoyuanzhou.github.io)**

---

## 目录结构

```
paper-notes/
├── index.html          # 首页导航
├── papers/             # 论文精读笔记（17 篇）
│   ├── cobra.html
│   ├── das.html
│   ├── diet.html
│   └── ...
├── topics/             # 专题学习笔记（12 个）
│   ├── notes-tokenizer.html
│   ├── notes-agent.html
│   ├── notes-rl.html
│   └── ...
└── code/
    └── tokenizer/      # Tokenizer 配套代码实现
        ├── 01_bpe.py
        ├── 02_wordpiece.py
        ├── 03_unigram_lm.py
        ├── 04_bbpe.py
        └── 05_tiktoken_guide.py
```

---

## 论文笔记 `papers/`

| 论文 | 关键词 |
|------|--------|
| COBRA | 推荐系统 |
| DAS | 推荐系统 |
| DIET | 推荐系统 |
| ETeGRec | 推荐系统 |
| GPSD | 推荐系统 |
| GRID | 推荐系统 |
| LIGER | 推荐系统 |
| LIGR | 推荐系统 |
| MTGRec | 推荐系统 |
| OneRec | 推荐系统 |
| OneSearch | 搜索 |
| Prefix-Ngram | NLP |
| SeMId | 推荐系统 |
| SId for Ranking | 推荐系统 |
| SIDE | 推荐系统 |
| TIGER | 推荐系统 |
| TRM | 推荐系统 |

---

## 专题笔记 `topics/`

| 专题 | 内容 |
|------|------|
| Tokenizer | BPE / WordPiece / Unigram LM / BBPE / tiktoken 原理与实现 |
| Agent | Agent 架构、ReAct、工具调用、多 Agent 协作 |
| RAG | 检索增强生成全流程 |
| ReAct | ReAct + Reflexion 范式 |
| RL | 强化学习基础 |
| Tool Calling | 工具调用格式与实现 |
| LLM Pipeline | 大模型训练推理流水线 |
| LLM Teams | 团队协作与工程实践 |
| Inference | 推理加速与优化 |
| Math | 数学基础 |
| GPU Hardware | GPU 硬件基础 |
| Harness Engineering | Harness Engineering 概念（2026） |

---

## 代码 `code/tokenizer/`

配套 Tokenizer 专题笔记的完整 Python 实现，从零实现各主流分词算法，注释翔实，可直接运行。

```bash
python code/tokenizer/01_bpe.py         # BPE 完整实现（训练+编码+解码）
python code/tokenizer/02_wordpiece.py   # WordPiece（BERT 风格）
python code/tokenizer/03_unigram_lm.py  # Unigram LM（T5 / XLNet）
python code/tokenizer/04_bbpe.py        # BBPE 字节级（GPT-4 / LLaMA3）

pip install tiktoken
python code/tokenizer/05_tiktoken_guide.py  # OpenAI tiktoken 完整使用指南
```

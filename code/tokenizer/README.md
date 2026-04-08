# Tokenizer 代码库

配套 `notes-tokenizer.html` 博客的完整 Python 代码实现，每个文件都有详尽注释。

## 文件说明

| 文件 | 算法 | 代表模型 | 内容 |
|------|------|----------|------|
| `01_bpe.py` | BPE（Byte Pair Encoding）| GPT-1/2、RoBERTa | 完整训练+编码+解码，含 Viterbi 对比 |
| `02_wordpiece.py` | WordPiece | BERT、DistilBERT | PMI 打分训练 + MaxMatch 推理 + 批量编码 |
| `03_unigram_lm.py` | Unigram Language Model | T5、XLNet、mBART | EM 训练 + Viterbi 分词 + Subword Regularization |
| `04_bbpe.py` | BBPE（字节级 BPE）| GPT-3/4、LLaMA3、Qwen | 字节级表示 + 中英混合演示 + 生僻字处理 |
| `05_tiktoken_guide.py` | tiktoken | GPT-3.5/4/4o | 完整使用指南，含 token 计费、截断、特殊 token |

## 快速运行

```bash
# 仅需标准库，无额外依赖
python 01_bpe.py
python 02_wordpiece.py
python 03_unigram_lm.py
python 04_bbpe.py

# 需要安装 tiktoken
pip install tiktoken
python 05_tiktoken_guide.py
```

## 各算法对比

```
BPE          → 贪心合并频率最高的 bigram，初始词表 = Unicode 字符
WordPiece    → 合并 PMI 最高的 bigram（更偏向"总是一起出现"的对），初始词表 = Unicode 字符
Unigram LM   → 从大词表剪枝，保留"删掉后语料概率损失最大"的 token
BBPE         → 初始词表 = 256 个字节（0-255），在字节上做 BPE，零 OOV
tiktoken     → OpenAI 的 BBPE 实现，Rust 编写，速度极快
```

## 主流模型用哪种

| 模型 | 算法 | 词表大小 |
|------|------|----------|
| GPT-2 | BPE | 50,257 |
| GPT-3.5 / GPT-4 | BBPE (cl100k) | 100,256 |
| GPT-4o | BBPE (o200k) | 200,019 |
| BERT | WordPiece | 30,522 |
| T5 | SentencePiece + Unigram | 32,000 |
| Gemini / PaLM | SentencePiece + BPE | 256,000 |
| LLaMA 1/2 | SentencePiece + BPE | 32,000 |
| LLaMA 3 | BBPE (tiktoken) | 128,256 |
| Qwen2 | BBPE (tiktoken) | 151,936 |

"""
tiktoken 完整使用指南（GPT-3.5 / GPT-4 / GPT-4o）
======================================
tiktoken 是 OpenAI 开源的 BPE tokenizer 实现，用 Rust 编写，Python 绑定。
是 GPT-3.5/4 系列的官方 tokenizer，速度比 HuggingFace tokenizers 快约 3-5 倍。

安装：pip install tiktoken

核心编码器：
  - cl100k_base：GPT-3.5 / GPT-4（词表 100,256）
  - o200k_base ：GPT-4o（词表 200,019）
  - p50k_base  ：GPT-3（词表 50,281）
  - r50k_base  ：GPT-2（词表 50,257）

GitHub：https://github.com/openai/tiktoken
"""

import tiktoken


# ═══════════════════════════════════════════════════════════
# PART 1：基础编解码
# ═══════════════════════════════════════════════════════════

def demo_basic():
    """演示基础的 encode/decode 功能。"""
    print("=" * 60)
    print("PART 1：基础编解码")
    print("=" * 60)

    # 加载编码器（第一次会从网络下载词表文件，之后缓存到本地）
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 词表

    # ── 基础 encode ────────────────────────────────────────
    text = "Hello, 你好世界！🌍"
    ids = enc.encode(text)
    print(f"\n输入文本: {text!r}")
    print(f"token ids: {ids}")
    print(f"token 数: {len(ids)}")

    # 看每个 token 对应什么字节
    print("\n逐 token 查看：")
    for token_id in ids:
        token_bytes = enc.decode_single_token_bytes(token_id)
        try:
            readable = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 跨字符边界的字节 token 无法单独解码
            readable = f'<bytes: {list(token_bytes)}>'
        print(f"  id={token_id:6d}  bytes={list(token_bytes):20}  text={readable!r}")

    # ── decode ─────────────────────────────────────────────
    decoded = enc.decode(ids)
    print(f"\n解码结果: {decoded!r}")
    print(f"往返一致: {'✓' if decoded == text else '✗'}")

    # ── 重要：空格位置会影响 token id！ ──────────────────────
    print("\n" + "-" * 40)
    print("空格位置影响 token id（tiktoken 的坑）：")
    tests = ["hello", " hello", "Hello", " Hello"]
    for t in tests:
        ids_t = enc.encode(t)
        print(f"  {t!r:12} → ids={ids_t}")
    # 结论："hello" 和 " hello" 是不同 token！
    # 拼接多段文本时要注意词间空格。


# ═══════════════════════════════════════════════════════════
# PART 2：不同编码器对比
# ═══════════════════════════════════════════════════════════

def demo_encoders():
    """对比不同 GPT 版本词表对同一文本的切分差异。"""
    print("\n" + "=" * 60)
    print("PART 2：不同编码器对比")
    print("=" * 60)

    encoders = {
        "r50k_base (GPT-2, 50K)": tiktoken.get_encoding("r50k_base"),
        "p50k_base (GPT-3, 50K)": tiktoken.get_encoding("p50k_base"),
        "cl100k_base (GPT-4, 100K)": tiktoken.get_encoding("cl100k_base"),
        # "o200k_base (GPT-4o, 200K)": tiktoken.get_encoding("o200k_base"),  # 需要更新的 tiktoken
    }

    # 几个有代表性的测试文本
    test_texts = [
        "1234567890",      # 数字：cl100k 会 3 位一组
        "你好世界",         # 中文：词表越大，切得越少
        "🎉🌍❤️",          # emoji
        "unforgettable",   # 英文长词
    ]

    print()
    for text in test_texts:
        print(f"文本: {text!r}")
        for enc_name, enc in encoders.items():
            ids = enc.encode(text)
            tokens = [enc.decode_single_token_bytes(i) for i in ids]
            try:
                token_strs = [b.decode('utf-8') for b in tokens]
            except UnicodeDecodeError:
                token_strs = [str(list(b)) for b in tokens]
            print(f"  {enc_name}: {len(ids)} tokens  {token_strs}")
        print()


# ═══════════════════════════════════════════════════════════
# PART 3：通过模型名自动选编码器
# ═══════════════════════════════════════════════════════════

def demo_encoding_for_model():
    """根据模型名自动匹配对应的编码器（推荐用法）。"""
    print("=" * 60)
    print("PART 3：通过模型名自动选编码器")
    print("=" * 60)

    model_names = [
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        # "gpt-4o",       # 需要较新版本的 tiktoken
        # "text-davinci-003",
    ]

    print()
    for model in model_names:
        try:
            enc = tiktoken.encoding_for_model(model)
            print(f"  {model:25} → {enc.name}  (词表大小: {enc.n_vocab})")
        except Exception as e:
            print(f"  {model:25} → 错误: {e}")


# ═══════════════════════════════════════════════════════════
# PART 4：实用工具 - 计算 Chat Messages 的 token 数
# ═══════════════════════════════════════════════════════════

def num_tokens_from_messages(messages: list, model: str = "gpt-4") -> int:
    """
    计算 OpenAI Chat API messages 列表的 token 数。

    为什么需要这个函数？
        Chat API 的每条消息不只是文本 token，还有格式开销：
        - 每条消息：+ 3 tokens（<|im_start|> role \n content <|im_end|>）
        - 如果有 name 字段：+ 1 token
        - 回复前缀：+ 3 tokens（<|im_start|>assistant）
        所以不能直接对 content 计 token 数。

    参考：OpenAI 官方文档 "How to count tokens with tiktoken"

    Args:
        messages: OpenAI Chat 格式的消息列表
                  [{"role": "user", "content": "..."}, ...]
        model: 模型名（用于选择正确的编码器）

    Returns:
        消耗的 token 总数
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 未知模型，用 gpt-4 的编码器作为近似
        encoding = tiktoken.get_encoding("cl100k_base")

    # gpt-3.5-turbo 和 gpt-4 的格式开销相同
    tokens_per_message = 3   # 每条消息的格式开销
    tokens_per_name = 1      # 有 name 字段时额外 +1

    n_tokens = 0
    for message in messages:
        n_tokens += tokens_per_message
        for key, value in message.items():
            n_tokens += len(encoding.encode(str(value)))
            if key == "name":
                n_tokens += tokens_per_name
    n_tokens += 3  # 回复前缀：<|im_start|>assistant

    return n_tokens


def demo_count_tokens():
    """演示如何计算 Chat 消息的 token 数。"""
    print("\n" + "=" * 60)
    print("PART 4：计算 Chat Messages token 数")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user",   "content": "Can you explain tokenization?"},
    ]

    n = num_tokens_from_messages(messages, model="gpt-4")
    print(f"\n消息列表（{len(messages)} 条）：")
    for msg in messages:
        role_tokens = len(tiktoken.get_encoding("cl100k_base").encode(msg["content"]))
        print(f"  [{msg['role']:10}] {msg['content']!r[:50]}  ({role_tokens} content tokens)")
    print(f"\n合计（含格式开销）：{n} tokens")
    print(f"注：GPT-4 上下文窗口 128K tokens，当前消耗 {n/128000*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# PART 5：实用工具 - 截断文本到指定 token 数
# ═══════════════════════════════════════════════════════════

def truncate_to_max_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4",
    ellipsis: str = "...",
) -> str:
    """
    把文本截断到不超过 max_tokens 个 token。

    为什么不直接截断字符串？
        字符数 ≠ token 数。中文 1 字符可能 = 1-2 token，
        英文单词 = 1-3 token。直接截字符可能截到 token 中间。
        用 token 截断才能精确控制 API 调用成本。

    Args:
        text: 待截断文本
        max_tokens: 最大 token 数
        model: 模型名
        ellipsis: 截断时末尾加的提示符

    Returns:
        截断后的文本
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    ids = enc.encode(text)

    if len(ids) <= max_tokens:
        return text  # 无需截断

    # 预留 ellipsis 的 token 数
    ellipsis_ids = enc.encode(ellipsis)
    truncated_ids = ids[:max_tokens - len(ellipsis_ids)]
    return enc.decode(truncated_ids) + ellipsis


def demo_truncation():
    """演示文本截断。"""
    print("\n" + "=" * 60)
    print("PART 5：文本截断到指定 token 数")
    print("=" * 60)

    long_text = (
        "Tokenization is a fundamental preprocessing step in natural language processing. "
        "It involves breaking down text into smaller units called tokens, which can be "
        "words, subwords, or characters. Different tokenization strategies have different "
        "trade-offs between vocabulary size, coverage of rare words, and sequence length. "
        "Modern large language models like GPT-4 use Byte Pair Encoding (BPE) at the byte "
        "level, which ensures that any Unicode text can be represented without out-of-vocabulary issues."
    )

    enc = tiktoken.get_encoding("cl100k_base")
    original_tokens = len(enc.encode(long_text))
    print(f"\n原始文本：{original_tokens} tokens")

    for max_t in [20, 50, 100]:
        truncated = truncate_to_max_tokens(long_text, max_tokens=max_t)
        actual_tokens = len(enc.encode(truncated))
        print(f"\n  max_tokens={max_t}（实际 {actual_tokens} tokens）：")
        print(f"  {truncated!r[:80]}...")


# ═══════════════════════════════════════════════════════════
# PART 6：特殊 token 处理
# ═══════════════════════════════════════════════════════════

def demo_special_tokens():
    """演示特殊 token 的处理方式。"""
    print("\n" + "=" * 60)
    print("PART 6：特殊 token 处理")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")

    print(f"\ncl100k_base 的特殊 token：")
    for token_str, token_id in enc.special_tokens_set.items() if hasattr(enc, 'special_tokens_set') else []:
        print(f"  {token_str!r} → id={token_id}")

    # 常用特殊 token：<|endoftext|>
    # 默认情况下，encode() 会报错如果输入包含特殊 token 字符串
    print("\n默认 encode（会对特殊 token 报错）：")
    safe_text = "Normal text without special tokens"
    ids = enc.encode(safe_text)
    print(f"  {safe_text!r} → {len(ids)} tokens")

    # disallowed_special=() 允许特殊 token 字符串作为普通文本处理
    print("\nencode(disallowed_special=())（特殊 token 当普通文本）：")
    text_with_special = "Hello <|endoftext|> world"
    ids_safe = enc.encode(text_with_special, disallowed_special=())
    print(f"  {text_with_special!r} → {ids_safe}")

    # allowed_special={'<|endoftext|>'} 允许特殊 token 生效
    print("\nencode(allowed_special={'<|endoftext|>'})（特殊 token 生效）：")
    ids_special = enc.encode(text_with_special, allowed_special={'<|endoftext|>'})
    print(f"  {text_with_special!r} → {ids_special}")
    print(f"  注意：<|endoftext|> 的 id = {enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})}")


# ═══════════════════════════════════════════════════════════
# PART 7：HuggingFace AutoTokenizer 对比用法
# ═══════════════════════════════════════════════════════════

def demo_huggingface_comparison():
    """
    演示 HuggingFace tokenizer 的对应用法（无需实际安装 transformers）。

    这里只展示代码模板，实际运行需要安装 transformers。
    """
    print("\n" + "=" * 60)
    print("PART 7：HuggingFace AutoTokenizer 对比用法（代码模板）")
    print("=" * 60)

    code = '''
# 安装：pip install transformers

from transformers import AutoTokenizer

# 加载（第一次会下载模型文件）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")   # WordPiece
# tokenizer = AutoTokenizer.from_pretrained("t5-base")           # Unigram LM
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")  # tiktoken

# 基础 encode
text = "Hello, tokenization world!"
# 方式 1：tokenize（看 token 字符串）
tokens = tokenizer.tokenize(text)
print(f"tokens: {tokens}")    # ['hello', ',', 'token', '##ization', 'world', '!']

# 方式 2：encode（只返回 ids）
ids = tokenizer.encode(text)
print(f"ids: {ids}")          # [101, 7592, 1010, ..., 102]  含 [CLS] 和 [SEP]

# 方式 3：__call__（返回完整字典，训练时最常用）
batch = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt",  # PyTorch tensor
)
print(f"input_ids shape: {batch.input_ids.shape}")
print(f"attention_mask : {batch.attention_mask}")

# decode
decoded = tokenizer.decode(ids, skip_special_tokens=True)
print(f"decoded: {decoded}")

# 批量编码
texts = ["Hello world", "Tokenization is fun"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
'''
    print(code)

    # tiktoken vs HuggingFace 对比
    print("\ntiktoken vs HuggingFace tokenizers 对比：")
    print(f"{'特性':20} {'tiktoken':25} {'HuggingFace tokenizers':25}")
    print("-" * 70)
    rows = [
        ("速度", "极快（Rust 实现）", "快（Rust 实现）"),
        ("支持模型", "OpenAI 系列", "几乎所有开源模型"),
        ("特殊 token", "手动处理", "自动处理 [CLS]/[SEP] 等"),
        ("返回格式", "List[int]", "dict（含 attention_mask）"),
        ("padding/truncation", "需手动实现", "内置支持"),
        ("主要用途", "计算 token 数/成本", "训练和推理"),
    ]
    for feature, tik, hf in rows:
        print(f"  {feature:20} {tik:25} {hf:25}")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    demo_basic()
    demo_encoders()
    demo_encoding_for_model()
    demo_count_tokens()
    demo_truncation()
    demo_special_tokens()
    demo_huggingface_comparison()

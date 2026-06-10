# Item Quantization 代码库

将 Item 的连续向量表征（embedding）离散化为**语义 ID（Semantic ID）**，
用于生成式推荐系统（Generative Recommender）中将 Item 作为 LLM 的 token 处理。

> 注：这与 LLM Tokenizer（BPE/WordPiece 等）处理**文本**的离散化不同——
> 这里的量化对象是**高维稠密向量**，目标是把连续空间映射为有限的离散码本。

## 文件说明

| 文件 | 方法 | 特点 | 依赖 |
|------|------|------|------|
| `01_rq_kmeans.py` | RQ-KMeans（残差量化 K-Means）| 轻量，无 GPU，可用于 RQ-VAE 热启动 | numpy, sklearn |
| `02_rq_vae.py` | RQ-VAE（EMA 残差量化）| EMA 在线更新 + 熵正则，GPU 加速，质量最高 | torch, numpy |
| `03_vq_vae.py` | VQ-VAE（向量量化）| 单层量化 + Dead Code Restart，结构简单 | torch, numpy, sklearn |

## 快速运行

```bash
# RQ-KMeans（只需 CPU）
pip install numpy scikit-learn
python 01_rq_kmeans.py

# RQ-VAE / VQ-VAE（支持 CPU / MPS / CUDA）
pip install torch numpy scikit-learn
python 02_rq_vae.py
python 03_vq_vae.py
```

## 三种方法核心对比

```
                RQ-KMeans          RQ-VAE             VQ-VAE
层数            L 层（残差）        L 层（残差）         1 层
每 Item token   L 个               L 个                1 个
码本大小        K（小）            K（小）              K（大，4096+）
可区分组合      K^L 种             K^L 种               K 种
码本更新        sklearn KMeans     EMA 在线更新         EMA 在线更新
防 Collapse     ✗                  熵正则               Dead Code Restart
GPU 支持        ✗                  ✅ MPS/CUDA          ✅ MPS/CUDA
推荐用途        快速原型/热启动     最终高质量码本        单 token 场景
```

## SID 输出格式

与 OneRec 论文（arXiv 2502.18965）对齐：

```
<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>
              └ 第1层 ┘└─ 第2层 ─┘└─ 第3层 ─┘
```

## 推荐使用流程

```
Step 1: 提取 Item embedding（BERT/SBERT 等文本编码器）
Step 2: RQ-KMeans 训练初始 codebook（快速）
Step 3: 用 init_from_rqkmeans() 热启动 RQ-VAE
Step 4: RQ-VAE 在线精炼 codebook（几个 epoch）
Step 5: encode() 获取全量 Item 的 codes [N, L]
Step 6: 格式化为 SID 字符串，写入 LLM 训练数据
```

## 参考论文

- RQ-VAE: [arXiv 2203.01941](https://arxiv.org/abs/2203.01941) "Autoregressive Image Generation using Residual Quantization"
- VQ-VAE: [arXiv 1711.00937](https://arxiv.org/abs/1711.00937) "Neural Discrete Representation Learning"
- OneRec: [arXiv 2502.18965](https://arxiv.org/abs/2502.18965) §2.1 Balanced Identifier Quantization
- 开源实现: [CarterHe479/Semantic-ID](https://github.com/CarterHe479/Semantic-ID)

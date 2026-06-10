Step 1 → 判别式打分：

$$p(y \mid \mathbf{f}, u)$$

Step 2 → 贝叶斯公式展开：

$$p(y \mid \mathbf{f}, u) = \frac{p(\mathbf{f} \mid y, u) \cdot p(y \mid u)}{p(\mathbf{f} \mid u)}$$

Step 3 → 分母 $p(\mathbf{f} \mid u)$ 是归一化常数，排序时可省略：

$$p(y \mid \mathbf{f}, u) \propto p(\mathbf{f} \mid y, u) \cdot p(y \mid u)$$

Step 4 → $p(y \mid u)$ 对所有候选 Item 相同，进一步省略：

$$\text{按判别式分数排序} \equiv \text{按 } p(\mathbf{f} \mid y, u) \text{ 排序}$$

Step 5 → 链式法则展开 $p(\mathbf{f} \mid y, u)$：

$$p(\mathbf{f} \mid y, u) = \prod_{k=1}^{n} p(f_k \mid f_{<k}, y, u)$$

Step 6 → 上式对应自回归解码 Item 特征序列：

$$\text{GR（完整特征覆盖）} \equiv \text{DR}$$

Step 7 → 但 SID 把丰富特征压缩成了紧凑离散码，结构化属性被丢失：

$$\text{实际 GR 表达力} < \text{DR} \quad \text{（差距来自特征覆盖度缺口）}$$

Step 8 → CoA 补回关键属性 $\mathbf{a} = (a_1, \ldots, a_m)$：

$$p(\mathbf{s} \mid u) = p(\mathbf{a} \mid u) \cdot \prod_{l=0}^{L-1} p(s_l \mid \mathbf{a}, s_{<l}, u)$$

Step 9 → 信息论严格保证：属性条件化降低每步不确定性

$$\Delta H_l = H(s_l \mid s_{<l}, u) - H(s_l \mid \mathbf{a}, s_{<l}, u) = I(\mathbf{a}; s_l \mid s_{<l}, u) > 0$$

$$\text{每步信息熵严格下降} \Rightarrow \text{beam search 更稳定} \Rightarrow \text{错误逐层衰减}$$
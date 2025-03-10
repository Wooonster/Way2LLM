# Quantization

按照 *量化发生的步骤* 区分，可以划分为 **PTQ（训练后量化，或离线量化）** 和 **QAT（训练感知型量化，或在线量化）**。
- PTQ (Post-Training Quantization) 量化：
	- is a straightforward technique where the weights of an *already trained* model are converted to lower precision *without necessitating any retraining*. Although easy to implement, PTQ is associated with *potential performance degradation*.
	- data-free：*不使用数据集*进行校准直接计算量化因子
	- calibration：会根据*少量真实数据进行统计分析*并*对量化因子进行额外校准*，但*耗费的时间更长*
- QAT (Quantization-Aware Training) 量化：
	- incorporates the weight conversion process *during the pre-training or fine-tuning stage*, resulting in enhanced model performance. However, QAT is *computationally expensive* and demands representative training data.
	- 会先在待量化的算子上增加一个*伪量化结构*，并在*训练时模拟量化过程并实时更新计算量化因子*（类似反向传播过程）及*原始权重*。QAT 由于较为复杂一般作为辅助措施存在，用于改进 PTQ 量化的技术手段。


按照 *量化方法* 可以划分为**线性量化**、**非线性量化（如对数量化）** 等多种方式，目前较为常用的是**线性量化**。其中线性量化又可以按照 *对称性* 划分为 **对称量化 (absolute maximum quantization)** 和 **非对称量化 (zero-point quantization)**，非对称量化为了*解决 weight分布不均匀* 问题，其在公式中增加了 `zero_point` 项：```math q_{weight}=round(weight/scale + zero_point)```，*使稠密数据部分可以得到更宽泛的数值范围*。
![(a)symmetric_quant](./media/(a)symmetric_quant.png)

## PTQ 训练后动态量化

##### 为什么量化对神经网络精度影响不大？

1. 因为一般权重和输入都经过 normalization，基本数据范围差距不大
2. 其次，因为激活函数的存在，数据影响会被平滑处理
3. 最后，因为大多数

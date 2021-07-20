---
published: true
title: SIM(Search-based Interest Model)论文解读
category: 推荐系统
tags: 
  - 推荐系统
  - CTR预估
  - 长期用户兴趣建模
layout: post
---

本文主要介绍 Alibaba 在 2020 年提出的可对超长用户行为序列进行建模的 SIM 模型，并对相关实验和其 online serving 进行分析与讨论。

论文链接：[Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction](https://arxiv.org/pdf/2006.05639.pdf)

# 相关工作

充分利用用户行为数据对点击率预估任务（Click-Through Rate Prediction）具有重要价值，而对用户的长期兴趣进行建模更是受到了工业界和学术界的广泛关注。

Alibaba 在 2017 年提出的 CTR 预估模型 DIN（Deep Interest Network）基于这样一个理念：用户的兴趣是多样的，且面对不同 candidate item 的时候是存在变化的，故 DIN 的关键思想是从用户行为中搜索有效信息，以建模在面对不同 candidate item 时用户具体的兴趣。具体来说，DIN 利用 attention 机制对用户历史行为进行了不同的加权处理，针对不同的 candidate item，用户历史行为的权重不一致。DIN 确实为利用用户行为数据进行 CTR 建模带来了很大的改进，但是面对长序列用户行为数据，DIN 的搜索公式产生的计算和存储成本是不可接受的。

MIMN，由 Alibaba 在 2019 年提出的一个基于 memory network 的模型，是第一个可以对长度达到 1000 的用户行为序列进行建模的工业解决方案。具体来说，MIMN 通过 UIC module 增量地将一个用户的多个兴趣 embed 到一个固定大小的记忆矩阵中，新的用户行为进入会导致该矩阵的更新。上述方式将用户建模的计算与 CTR 预估分离，因此对于 online serving，latency 不会成为瓶颈，并且存储成本只取决于记忆矩阵的大小，而该大小远小于原始行为序列。然而，当用户行为序列的长度增加到10倍及以上时，MIMN 无法精确捕捉对于特定 candidate item 用户兴趣所在，这主要是因为：随着用户行为大幅增长，将所有历史行为编码到固定大小的记忆矩阵中会导致记忆单元中包含大量噪声。而且，MIMN 在 memory network 中丢弃了对于用户兴趣建模很重要的 candidate item 信息。

 那么，我们是否可以应用和 DIN 类似的搜索技巧并设计一种更有效的方法来从长序列用户行为数据中提取知识？本文的 SIM 解决的就是这样一个问题，下面我们看看 SIM 模型具体是怎样设计的。

# 模型

SIM 是一个两阶段的基于搜索的 Interest Model，相对应地设计了两个搜索单元：通用搜索单元 General Search Unit（GSU）和精确搜索单元 Exact Search Unit（ESU）。GSU 从超过一万个用户行为中寻找最相关的 $K$ 个行为输出给 ESU，ESU 则利用多头注意力来捕捉不同的用户兴趣，然后模型遵循传统的 Embedding&MLP 范式，以ESU输出的精确的长时间用户兴趣和推荐系统常用的其他特征作为输入进行训练。下面是模型框架的示意图。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/0.png" alt="0" style="zoom:80%;" />

接下来我们具体看看 GSU 和 ESU 的设计思想和具体结构。

### GSU

给定一个 candidate item（CTR 模型打分的目标），只有一部分用户行为是有价值的。 这部分用户行为与最终用户决策密切相关，而 GSU 就用于挑选出这些相关的用户行为来帮助用户兴趣建模，同时减少资源占用和响应延迟（相对于使用整个用户行为序列）。那么具体如何挑选呢？GSU 会为长期用户行为序列里的每个用户行为都计算一个和 candidate item 的相关性分数，然后基于此分数选择 $K$ 个最相关的构成子行为序列作为 ESU 的输入。模型提出了两种 GSU 的实现方式：Hard Search 和 Soft Search，二者不同之处就是相关性分数的计算方式，公式如下：

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/1.png" alt="0" style="zoom:50%;" />

Hard Search 不需要深度模型，它选择与 candidate item 属于同一类别的行为，并将其聚合为子行为序列发送给 ESU，公式中的 $C_a$ 和$C_i$ 分别表示 candidate item 和第 $i$ 个行为 $b_i$ 的类别。 硬搜索简单而有效，非常适合在线服务，后面我们会更加深入地讨论这一点。

Soft Search 则通过 candidat item 和 behavior 向量的内积计算相关性分数，然后将每个行为 $e_i$ 通过其相关性分数 $r_i$ 加权求和得到行为序列的整体表示 $U_r$，然后 $U_r$ 和目标向量 $e_a$ 会被 concat 起来作为后续 MLP（多层感知）的输入。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/2.png" alt="0" style="zoom:40%;" />

整个 Softmax Search 模块的训练是一个辅助 CTR 预估任务。这里引入了辅助训练，而没有直接利用第二阶段中从短期用户兴趣建模中学到的参数，作者的解释是：长期和短期数据的分布是不同的，在 Soft Search 中直接使用从短期用户兴趣建模中学到的参数可能会误导长期用户兴趣建模。

### ESU

ESU 以 GSU 输出的子行为序列作为输入，对其与 candidate item 的关系进行精确建模。它是一个注意力模型，可基于 DIN 或者 DIEN 实现。

考虑到子序列里的用户行为很可能跨越了很长时间，导致各用户行为的贡献程度不同，模型为每个行为引入了时序的表征。具体来说，用户行为 $b_j$ 与 candidate item 之间的时间间隔 $\Delta_j$ 会被编码为 $e_j^t$，然后和行为本身的 embedding $e_j^*$ concat 起来，用 $z_j$ 表示，公式里的 $z_b$ 为 $[z_1, ..., z_K]$ 整个序列的表示。和 Soft Search 类似，模型依然通过 candidate item 和每个行为间的 embedding 内积计算相关性，经过 softmax 转化为一个加和为 1 的 $K$ 维向量 $att_{score}$，第 $j$ 维对应 $b_j$ 和 candidate item 的相关程度，然后将带有时序信息的行为表征 $z_b$ 通过 $att_{score}$ 加权求和得到行为序列的整体表示 $head$。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/3.png" alt="0" style="zoom:50%;" />

因为模型用到了多头注意力，所以会有多个 $head$，可以在不同向量空间捕获用户的不同兴趣，所有 $head$ 会被 concat 起来作为最终的用户长期兴趣表征，作为 ESU 的输出，和其他推荐系统常用的其他特征（如短期用户行为、用户画像、上下文信息等）作为 MLP 的输入，进行 CTR 预估。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/4.png" alt="0" style="zoom:40%;" />

GSU 和 ESU 在交叉熵损失函数下共同训练，同时二者中的 embedding 参数也是共享的。 $α$ 和 $β$ 是控制损失权重的超参数，在文中的实验，如果 GSU 使用 Soft Search，$α$ 和 $β$ 都设置为 1，使用 Hard Search 的 GSU 是非参数的，$α$ 设置为 0。

# 实验

实验在 Amazon(Book) 和 Taobao 等两个公开数据集和一个工业数据集上进行。 其中，Amazon 数据集由来自 Amazon 的产品评论和 metadata 组成，文中使用的是 Amazon 数据集的 Books 子集，包含 75053 个用户、358367 个项目和 1583 个类别；Taobao 数据集是来自淘宝推荐系统的用户行为集合，该数据集包含点击、购买等多种类型的用户行为，包含约 800 万用户的用户行为序列；工业数据集的样本来源于 Alibaba 在线展示广告系统，其历史行为序列以 14 天为界限划分长期和短期行为。工业数据集中超过 30% 的样本包含长度超过 10000 的序列行为数据，此外，行为序列最大长度达到54000，是 MIMN 中的 54 倍。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/5.png" alt="0" style="zoom:50%;" />

上表是不同模型在公开数据集上的实验结果对比。由于原生 DIN 只接受短期用户行为特征作为输入，为了比较模型在长期用户兴趣建模上的表现，Avg-Pooling Long DIN 对长期行为进行平均池化，并将获得的 embedding 与其他特征 embedding concat 起来，再利用 DIN 进行训练。 

从表中可以看出：与 DIN 相比，其他考虑长期用户行为特征的模型表现要好得多，这表明长期用户行为有助于 CTR 预测任务。SIM 优于所有其他长期兴趣模型，这有力地证明了其两阶段搜索策略对于长期用户兴趣建模是有用的（消融实验也对此进行了进一步验证），此外，引入长期行为的时序信息确实达到了进一步的优化。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/6.png" alt="0" style="zoom:50%;" />

上表是消融实验结果，Avg-Pooling without Search 即 Avg-pooling Long DIN 一样，只使用平均池化处理长期行为，没有搜索的过程。 $Only First Stage$ 两个模型，都维持 SIM 的 GSU 不变，而将 ESU 替换为平均池化操作。

从表中可以看出：与简单平均池化相比，所有对长期行为进行过滤的方法都极大地提高了模型性能， 这表明原始长期行为序列中确实存在大量噪声，可能会破坏长期用户兴趣学习；与只有一级搜索的模型相比， 在第二阶段引入基于注意力的搜索为模型取得了进一步进展，这表明精确建模用户对 candidate item 的不同长期兴趣有助于优化 CTR 预估结果；引入长期行为的时序信息达到了进一步的优化，这表明不同时期用户行为的贡献不同。 

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/7.png" alt="0" style="zoom:50%;" />

文章还定义了一个有趣的指标：$Days till Last Same Category Behavior (d_{category})$，表示用户此次点击和上次点击同类别 item 的间隔天数，并以此来衡量模型对于短期或长期兴趣的选择偏好。A/B 实验之后，作者基于 $d_{category}$ 分析了来自 SIM 和 DIEN 的点击样本，其分布如上图所示，可以发现两个模型在短期行为上（$d_{category}$ < 14）几乎没有区别，而从长期来看，SIM 所占的比例更大。 

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/8.png" alt="0" style="zoom:40%;" />

此外，上表显示了两个模型在工业数据集上 $d_{category}$ 平均值和用户在 target item 上具有同类别历史行为的概率的统计结果，证明 SIM 确实能更好地进行长期兴趣建模的结果，与 DIEN 相比，SIM 更喜欢推荐与人们长期行为相关的项目。

# 在线服务

下表展示了不同模型在工业数据集上的表现。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/9.png" alt="0" style="zoom:40%;" />

可以看到，虽然 Soft Search 确实比 Hard Search表现更好，但二者之间只有微小的差距，而 Soft Search 需要更多计算和存储资源。此外，对于两种不同的搜索策略，作者提到他们对超过 100 万个样本和 10 万个具有长期历史行为的用户进行了统计，结果表明，Hard Search 保留的用户行为可以覆盖 Soft Search 保留用户行为的 75%。于是，对于在线服务，在效率和性能之间的权衡之下选择了更简单的 Hard Search GSU。下面是整个基于 SIM 的实时预测（Real-Time Prediction，RTP）系统流程图。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/10.png" alt="0" style="zoom:70%;" />

该系统为每个用户构建了一个两级结构化索引，命名为用户行为树（UBT）。UBT 遵循 Key-Key-Value 的数据结构：第一个键是用户 id，第二个键是类别 id，最后一个值是属于每个类别的行为。 

UBT 是一个分布式系统，能够灵活地提供高吞吐量查询，而且 GSU 的 UBT 索引可以离线预建，因此 Hard Search 只需要从离线内置的两级索引表中进行搜索，用户行为的长度便从上万条减少到上百条，从而释放在线系统中长期行为的存储压力。在线系统中 GSU 的响应时间非常短，与 GSU 的计算相比可以省略，此外，其他用户特征也可以并行计算。Hard Search 有效且系统友好，故线上模型从 MIMN 转换到 SIM 后，不同吞吐量下的延迟增大量是完全可接受的。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/推荐系统（一）/11.png" alt="0" style="zoom:50%;" />

SIM (hard) with timeinfo 相对于 MIMN（前一个 product model），离线实验 AUC 增益为 0.008，在淘宝首页猜你喜欢栏目的 A/B 实验中， 提升了 7.1% 的 CTR 和 4.4% 的 RPM。 随后，该模型上线部署，每天服务于淘宝主场景流量，对于业务收入增长显著。
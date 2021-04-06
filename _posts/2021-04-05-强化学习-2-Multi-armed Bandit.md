---
published: true
title: 多臂老虎机（Multi-armed Bandit, MAB)
category: 强化学习
tags: 
  - 强化学习
  - 概率论
  - MAB
  - UCB
  - Thompson Sampling
layout: post
---

本文首先对 MAB 问题进行介绍，然后讲解其两种主要解决方法：Upper Confidence Bound (UCB) 和 Thompson Sampling。

# MAB

Multi-armed Bandit 是强化学习中关于 Exploration & Exploitation 的一个经典问题。最经典的 MAB 场景为：一个人进了一家赌场，面前有 $N$ 台老虎机，假设每台老虎机都有一定概率吐钱或者不吐钱（每台的概率不太一样），在事先不知道每台老虎机的真实盈利情况以及最多摇 $T$ 次老虎机的限制下，此人需要根据积累的经验来选择下次摇哪台老虎机或者停止赌博，以最大化自己的累计收益。

最简单的 MAB 假设环境不会变化，但在实际情况中，这种假设经常不成立。比如，在一位玩家已经发现某台机器盈利概率很高并一直摇它之后，赌场可能人为降低这台机器吐钱的概率，这里 MAB 问题的环境就是随着时间/玩家的行为会发生变化。

另外，MAB 有一类很重要的变种叫做 Contextual MAB，几乎所有在线广告/视频推荐都可以看成是 Contextual MAB 问题。在这类问题中，每个 arm 的回报会和当前用户的特征（也就是这里说的 context）有关。那么，推荐系统中的 MAB 问题可以抽象为：我们有 $N$ 个视频（商品），在事先不知道用户 $u$ 对 $N$ 个视频（商品）的偏好情况下，我们要尽可能推荐用户更喜欢的视频（商品），以使得视频被用户 $u$ 观看（或者商品被用户 $u$ 购买，后面我们统称为物品被用户 $u$ 转化）。

每个物品 $i$ 被用户 $u$ 转化的事件，对应一个 Bernoulli 分布：

$$
f(k\ ;\ p_i) =
\begin{cases}
p_i,\ k = 1 \\ 
1 - p_i,\ k = 0
\end{cases}
$$

我们将 $p_i$ 叫做物品 $i$ 的被转化率，也就是对于每个物品 $i$ ，用户 $u$ 转化的概率为 $p_i$。每个物品的被转化率事先是未知的，而为了估计被转化率，通常有两种做法：UCB 和 Thompson Sampling。

# UCB

UCB，全称为Upper Confidence Bound。在 MAB 问题下，UCB 其实就是在某个置信度下，取置信区间的上界作为估计。

$$
\begin{aligned}
&\{ Item_i,\ reward \} \to Bernoulli(p_i), \\ \\
&Empirical \ risk = || \frac {reward_i} {round_i} - p_i ||
\end{aligned}
$$

Hoeffding 不等式是 UCB 中的核心概念，给出了置信度和置信区间的关系，其证明：

$$
P(Empirical \ risk >= \ \delta) \ <= \ 2 * e^{-2*round_i*\delta^2}
$$

我们构造 $\delta = \sqrt {\frac {\ln round}{round_i}}$，其中 $round$ 表示推荐所有物品或拉所有老虎机的次数，那么置信度就是 $1 - \frac {2}{round^2}$ 。当然，我们也可以给 $\delta$ 乘以实数，比如 $\delta = \sqrt {\frac {2 \ * \ \ln round}{round_i}}$ ，这样置信度就是 $1 - \frac {2}{round^4}$ 。

由以上公式可看出：$\delta$ 与 $(round_i)^{-\frac {1}{2}}$ 成正比，所以对于被推荐次数少的物品 $i$，$\delta$ 更大，用 UCB 方法估计的被转化率就可能更大，即较容易被优先推荐；此外，对于任意物品 $i$， 随着试验次数的增加，实证误差的置信度将逐渐接近1，即其被转化率的估计会越来越准；最后，随着试验次数的增加，$\delta$ 逐渐减小，对应物品的被转化率估计的置信区间都会缩小，也就是估计的值会越来越接近真实值。

# Thompson Sampling

这里，我们先了解下共轭先验的概念：在贝叶斯统计中，如果一个似然函数的后验分布与先验分布属于同类，则先验分布与后验分布被称为共轭分布，而先验分布被称为此似然函数的共轭先验（Conjugate Prior），比如，高斯分布家族在高斯似然函数下与其自身共轭（自共轭）。

共轭分布的意义在于：因为后验分布和先验分布形式相近，只是参数有所不同，这意味着当我们获得新的观察数据时，我们就能直接通过参数更新，获得新的后验分布，此后验分布将会在下次新数据到来时成为新的先验分布。如此一来，后验分布的更新就不再需要大量的计算，十分方便。

对于 MAB，Bernoulli 分布正好有 Beta 分布作为共轭先验（对于 Beta 分布不了解的同学，可以参考 [如何通俗理解 Beta 分布](https://www.zhihu.com/question/30269898) 中的回答）。准确来说，每次选出推荐的物品后，根据物品是否被用户转化来更新 Beta 分布的参数（转化时 $\alpha$ 加1，否则 $\beta$ 加1），有了新参数的 Beta 分布就是结合了这次经验后，对被选中物品的被转化率更准确的估计。

$$
\begin{aligned}
&Beta(\alpha,\ \beta) = \frac{1}{B(\alpha,\beta)}x\ ^{\alpha - 1}(1-x)^{\beta-1},\ \ \alpha,\beta > 0, \\ \\
&E(X) = \frac{\alpha}{\alpha + \beta}, \\ \\
&Beta(\alpha_0,\ \beta_0) + Sample\ Data = Beta(\alpha_0 + Positve,\ \beta_0 + Negative)
\end{aligned}
$$

Thompson Sampling 的具体步骤如下：

1. 为每个物品 $i$ 预估一对 Beta 分布的参数 $\alpha$，$\beta$；
2. 每次试验前从每个物品 $i$ 的 Beta 分布中随机采样获得对应的被转化率 $p_i$；
3. 选择被转化率最大的物品 $i_{p.max}$；
4. 根据用户对物品 $i_{p.max}$ 是否转化来更新该物品的 Beta 分布参数：转化则 $\alpha$ 加1，否则 $\beta$ 加1；
5. 循环 2~4 步 $T$ 轮。

Thompson Sampling 允许我们根据实际情况，针对每个物品初始化不同的 Beta 分布参数，故相比于 UCB，Thompson Sampling 更大程度地利用了每个物品被转化率的先验知识。
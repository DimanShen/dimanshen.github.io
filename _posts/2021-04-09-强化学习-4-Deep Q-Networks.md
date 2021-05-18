---
published: true
title: Deep Q-Networks
category: 强化学习
tags: 
  - 强化学习
  - Q-Learning
  - Deep Q-Network
layout: post
---

本文向大家介绍 Deep Q-Network 的产生及发展。

# 从 Q-Learning 到 DQN

### 传统 Q-Learning 的局限性

我们在  [Q-Learning](https://dimanshen.github.io/强化学习/2021/04/06/强化学习-3-Q-Learning/) 一文中向大家介绍了基于 Q-Table 的 Q-Learning，实际上这种传统 Q-Lerning 存在很多局限性：首先，在面对复杂场景如大型游戏时，其状态空间十分庞大，此时维护一个 Q-Table 会变得十分低效；其次，传统 Q-Learning 不是 end-to-end 的学习，状态和动作这些特征是手工定义的，于是算法的 performance 很大程度地依赖于特征设计的质量。当面对复杂问题，人类经验不足甚至错误时提取的 feature 肯定无法达到很好的效果；再就是不同任务的特征不同，比如射击类游戏的特征应该是目标位置、远近和射击角度等等，而换一个游戏，如 flappy bird 就需要花很多时间设法获取小鸟的位置以及各个水管的位置等，这会导致一个场景的算法不能很好地迁移应用到另一个场景上，没有通用性。

### 从深度学习看 Q-Learning

针对上面种种局限性，有人提出了用神经网络替代 Q-Table 来实现 Q-Learning。利用深度神经网络近似 Q-Learning 中的 Value Function，再通过迭代的训练让近似更加准确，可以解决维护q-table效率底下的问题；此外，利用 CNN 天然适合处理图像的特性，通过用 CNN 处理游戏画面，可以免去手工设计特征的步骤。这样，模型自己去考虑应该提取什么样的 feature，我们甚至不关心最后具体学到什么，只需要模型能在各种状态下做出正确的动作即可，于是也解决了可靠性和通用性的问题。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/0.png" alt="0" style="zoom:80%;" />

但要将强化学习与深度学习结合，也会面临一些新的挑战。首先，单帧的静态画面无法展示出动作的趋势，如何处理由图像带来的这种时序限制问题呢？其次，深度学习的成功往往依赖于大量有标签的样本来进行有监督学习，而增强学习只有一个 reward 作为返回值，并且通常是稀疏的，还带有噪声和延迟；此外，大多数深度学习算法会假定数据样本独立，而强化学习中的状态却是相关的，序列中前后状态会相互影响；最后，通常深度学习的目标分布是固定的，比如做物体检测，一张图片是房子就是房子，不会改变变，但在强化学习中，目标会随着算法学习新行为而发生变化，比如超级玛丽，前面的场景和后面的场景不一样，可能前面的场景训练好了，后面场景却并不适用。

### DQN

论文链接：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，DeepMind，2013

Deep Mind 在 2013 年发表了《Play Atari with Deep Reinforcement Learning》，通过将图片帧序列作为模型输入和 Experience Replay 方法解决了上面提到的前 3 个问题并部分解决了第 4 个问题。这篇 paper 是 Deep Reinforcement Learning 的开山之作，论文将 CNN 和 Q-Learning 结合，把提出的模型叫做 Deep Q-Network，简称 DQN。论文将 DQN 应用在了 7 个 Atari 游戏中，有 6 个游戏最终效果比以往的方法都要好，其中 3 个还超过了人类玩家的水平。

我们来具体看看这篇 paper 是如何解决这 4 个问题的。首先，这篇论文里采用堆叠的 4 帧作为 CNN 的输入解决了时序限制的问题，单帧无法看出动作趋势，但连续的多帧是可以的。另外，论文提出了 Experience Replay 的概念来处理余下的问题。Experience Replay 指的是创建一个 “replay buffer”，即缓冲区，来存储 Agent 与环境交互的过往经历，然后从中随机采样进行训练。与传统 Q-Learning 中交互与更新并行的方式相比，Experience Replay 有如下好处：首先，连续样本之间有很强的关联性，使用连续样本会使训练效率低下，且容易陷入局部最优，而随机的样本能打破这种关联；其次，Agent 的行为分布会在其先前状态下进行平均，可以平滑学习过程、避免参数出现波动或发散。如果不使用 Experience Replay 而采用 Online Learning 的机制，那当前的参数就决定了下一个训练样本，而我们又要根据这个样本训练我们的参数，这样很容易导致模型收敛到一个局部最优解，甚至产生灾难性的偏移结果；最后，Experience Replay 使得每一步经验有被多次用于神经网络权重更新的可能，这样数据的使用更有效率，同时也可以避免遗忘以前的经验。

- **模型结构**

  DQN 使用堆叠的相邻 4 帧作为输入，其首先对每帧都会做灰度、下采样和裁剪等预处理进行降维，然后用 CNN 提取图片中的状态和动作等空间信息，最后通过全连接层来输出给定状态下可能发生的每个动作的 $q$ 值，最大的 $q$ 值就对应着最佳的动作。最开始的时候，DQN 模型参数和 Q-Table 一样，也是随机初始化的，此时输出的 $q$ 值并不可靠，但随着迭代更新，模型参数会得到优化，对每个状态下的最佳动作选择也会越来越准确。

  <img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/1.png" alt="0" style="zoom:40%;" />

  回到上一小节 Q-Learning 和 DQN 的对比示意图，大家可能会疑惑：为什么 DQN 使用单独的状态作为模型输入，而不像 Q-Learning 一样使用状态-动作对呢？这主要是考虑到效率问题。如果使用状态-动作对作为输入，则需要对每个动作都单独地进行前向计算，从而模型参数会随动作数呈线性比例增长，而 DQN 目前这种结构在给定状态下只需要一次前向计算，就能够为所有 action 计算 $q$ 值。

- **算法流程** 

  <img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/2.png" alt="0" style="zoom:40%;" />

  DQN 算法涉及 3 个公式，公式（1）是用 Q-Learning 和 Bellman 方程计算目标 $q$ 值；公式（2）是神经网络的 loss function，即模型的优化目标；DQN 采用随机梯度下降进行训练和更新，公式（3）是梯度更新公式，$\rho(s, a)$ 是序列 $s$ 和动作 $a$ 上的概率分布，我们称之为行为分布，在优化损失函数 $L_i(\theta_i)$ 时，$\theta_{i-1}$ 的参数是保持固定的。需要注意的是：因为我们用网络权重去估计预期回报，所以 RL 问题中的 target 或者说 label 是取决于网络权重的，这与监督学习中的 target 不同，其在学习开始之前就已经确定。

  <img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/3.png" alt="0" style="zoom:40%;" />
  
  上图是 DQN 的具体算法流程：先初始化 Replay Memory 的容器 $D$ 大小为 $N$，随机初始化神经网络的权重，然后开始 $M$ 局游戏。每局游戏初始状态为 $s_1$，经过预处理后得到 $\phi_1$，之后每一步操作遵循 $\epsilon$-greedy policy 选择动作，执行动作后增量获取的 reward 和 next state 信息在预处理后都存储到 $D$ 中，然后进入模型更新，从 $D$ 中随机选取 mini batch 并按照 Bellman 方程计算目标 $q$ 值，最后通过梯度下降减小 loss，来优化神经网络的权重。

# Deep Q-Learning 的发展

从 2013 年 DQN 被提出以来，在其上有了很多改进，下面列举比较 well-known 的几种，其中主要介绍 Separate Target Q-Network 和Double DQN。

### Separate Target Q-Network

论文链接：[Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)，DeepMind，2015

前面我们提到，强化学习的目标会随着算法学习新行为而发生变化，这是因为 target 取决于网络参数，模型使用相同的参数估计目标值和 $q$ 值。在训练中，$q$ 值移动的同时目标值也在移动，这会导致训练过程中的大幅振荡，就像农夫追赶移动的牛，最后追赶的轨迹可能十分曲折。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/4.png" alt="0" style="zoom:50%;" />

于是，在2015年，Deep Mind 在 Nature 上发表了 DQN 的改进版论文，即在原始的 DQN 中增加一个独立的目标网络，通过短时间固定 Q-Target 来缓解以上问题。具体来说，目标网络是在线网络的定期副本，每隔比如 1k 个 step，目标网络会复制在线网络的最新参数，然后在接下来的 1k 个 step，在线网络持续更新，但目标网络参数固定不变。如此循环，可以保证每段时间内 target 是固定不变的，相比于原始的 DQN，这样训练更加稳定。

### Double DQN

论文链接：[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)，DeepMind，2015

改进后的 DQN 仍然存在一个问题，即在估算 Q-Tagret 时，模型用相同的参数来选择和评价一个动作，$max$ 操作导致模型会更偏向于选择过分评估值，论文里叫做 overestimated values，这样最后得到的可能是次优的估计值。通俗地说，模型参数一开始是随机初始化的，$q$ 值存在很大噪声，因此我们直接认为最大的 $q$ 值对应最佳动作可能会导致 false positive，如果一开始某些动作获得了比最佳动作更高的 $q$ 值，随后我们又总是通过同样的参数去强化这种情况，模型的学习会变得很复杂。

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/5.png" alt="0" style="zoom:50%;" />

Double DQN 这篇论文针对这种情况提出了解决方案：使用两个网络计算 Q-Target，将动作选择与动作评价分离。具体来说，Double DQN 中计算 target 使用了两个不同的 $\theta$，分别来自在线网络和目标网络。在线网络负责通过 greedy policy 选择动作，而带有 delayed $\theta$ 的目标网络计算该动作对应的 Q-Target，从而更加公平地对 greedy policy 的选择进行评价。

至于 Double DQN 的实现，由于改进版 DQN 中已经引入目标网络，这给 Double DQN 提供了现成的模型框架，故 Double DQN 对目标网络的更新可以与改进版 DQN 保持不变，仍然是在线网络的定期副本，而只需要将改进版 DQN 的目标网络中对动作的选择从使用delayed $\theta$ 改成使用 online $\theta$ 即可，这种处理达到了计算开销的最小化。

### Dueling DQN

论文链接：[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)，DeepMind，2015

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/6.png" alt="0" style="zoom:40%;" />

Dueling DQN 将 Q-Function 的计算分成了状态价值和动作增益两部分。通过将二者解耦，模型可以更直观地了解哪些状态本身是有价值或不重要的，而不必了解每个动作在每个状态下的效果。

### Prioritized Experience Replay

论文链接：[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)，DeepMind，2015

PER 是对 Experience Replay 的改进，论文里认为那些导致估计和目标有更大差异的 experience 蕴含了更丰富的知识，即这样的 experience 更有价值，但其发生的频率可能比较低，所以应该针对性地给予更高的 priority。

PER 通过 Stochastic Prioritization 和 Importance Sampling（IS）来定义 experience 的优先级从而更改采样分布，其中 Stochastic Prioritization 是一种在 pure greedy prioritization 和 uniform random sampling 之前进行插值来定义优先级的方式，而 Importance Sampling 则主要用于 debias，对偏差进行退火。

### Rainbow

论文链接：[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)，DeepMind，2017

<img src="https://raw.githubusercontent.com/DimanShen/dimanshen.github.io/master/_posts/image/强化学习（四）/7.png" alt="0" style="zoom:30%;" />

Rainbow 这篇论文从标题就可以看出来，它是一些 DQN improvements 的组合，例如前面提到的 4 种改进在 Rainbow 模型里都有出现。通过消融实验的结果可以看到，Rainbow 模型效果也确实出类拔萃，感兴趣的话可以看看论文。
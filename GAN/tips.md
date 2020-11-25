# Data augmentation methods for machine-learning-based classification of bio-signals
+ 2017 10th Biomedical Engineering International Conference
+ 所提出的方法对于提高小数据集的预测精度特别有效。机器学习分类器。
+ 数据增强，四种方法（每种增强方法只适用于某些时期，而不适用于整个时期。）：
   + 全部数据位移指定时间量（移位范围为+10ms和-10ms，因此原始数据集增加了三倍。）
   + 放大所有时间数据；90%和110%，所以原始数据集是原来的三倍。
   + 移位近峰数据
   + 放大近峰数据
+ 数据集小时候分类提升比较明显

# Deep EEG super-resolution: Upsampling EEG spatial resolution with Generative Adversarial Networks
+ 2018 IEEE EMBS International Conference on Biomedical Health Informatics；ccf-c类
+ 数据集： BCI Competition III, Dataset V .32个样本，每个样本64个epoch,下采样16个通道，一个用作低分辨率数据（LR）一个用作高分辨率数据（HR）
+ 通过生成通道级上采样数据来有效地内插大量丢失的通道与基线双三次插值方法相比，我们提出的GaN模型的均方误差(MSE)和平均绝对误差(MAE)分别降低了104倍和102倍。我们通过在原始分类任务上训练分类器来进一步验证我们的方法，在使用超分辨数据时，分类器的准确率损失最小。
+ 图像超分辨率(SR)：低分辨率数据生成高分辨率数据（4倍新图像《Generative Adversarial Nets》 Advances in Neural Information Processing Systems, vol. 27, pp. 2672-2680, 2014.
+ 稳定性问题，选择WGANs，比原始GAN的算法实现流程改了四点：
   + 判别器最后一层去掉sigmoid
   + 生成器和判别器的loss不取log+ 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
   + 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
+ 结论，比双三次插值相比，WGAN的均方误差和最大均方误差分别降低了约104倍和~102倍。2 4规模的数据分类精度分别下降4%，9%。网络结构清楚，复现比较容易

# Generating target/non-target images of an RSVP experiment from brain signals in by conditional generative adversarial network
+ 2018 IEEE EMBS International Conference on Biomedical Health Informatics；ccf-c类
+ 重建视觉图像，一个思想Brain2Image_Converting_Brain_Signals_into_Images
+ 目标图像 非目标图像
+ DCGAN的discriminator提取到的图像特征更有效，更适合用于图像分类任务。且训练稳定
   + 使用卷积和去卷积代替池化层
   + 在生成器和判别器中都添加了批量归一化操作
   + 去掉了全连接层，使用全局池化层替代
   + 生成器的输出层使用Tanh 激活函数，其他层使用RELU
   + 判别器的所有层都是用LeakyReLU 激活函数
+ acc 62


# Improving brain computer interface performance by data augmentation with conditional Deep Convolutional Generative Adversarial Networks
+ arXiv.org
+ 提出cDCGAN+CNN，和上一篇一样。（公式推导）
+ 将DCGAN扩展到带有训练数据标签的条件版本，并生成带标签的人工脑电信号。
+ CNN分类。原始EEG训练数据(1*RAW)、人工EEG数据(1*人工)和混合EEG数据(0.5*RAW+0.5*人工)的分类准确率几乎相同，分别为82.86%、82.86%和82.14%。按倍数混合后，精度明显增加

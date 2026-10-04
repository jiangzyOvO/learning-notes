# 第十三讲专题：ELBO、KL、重参数化与完整标量 VAE

从潜变量积分出发，推导 ELBO、两种 KL 与重参数化，并实现标量线性高斯 VAE。前置：[概率与自回归](13-probability-autoregressive-deep-review.md)。

## 1. 三种分布与一个难算积分

p(z)是先验，p_theta(x|z)是解码器，q_phi(z|x)是编码器提供的近似后验。真实后验p_theta(z|x)由生成模型确定，一般不等于编码器分布。高维非线性模型的p_theta(x)=积分p_theta(x|z)p(z)dz通常难直接计算，因此优化可计算下界。

## 2. 从Bayes关系得到ELBO

固定x，Bayes关系给出log p(x)=log p(x,z)−log p(z|x)。在q(z|x)下取期望并加减E_q log q：

```text
log p(x)
 = E_q[log p(x,z) - log q(z|x)]
   + E_q[log q(z|x) - log p(z|x)]
 = ELBO + KL(q(z|x) || p(z|x))
```

第二项KL非负，所以ELBO≤log p(x)。符号||表示按前一分布取期望的对数比，不是范数；KL通常不对称，不是欧氏距离。等号在q等于真实后验时成立。

再拆联合p(x,z)=p(x|z)p(z)：

```text
ELBO = E_q log p(x|z) - KL(q(z|x) || p(z))
负ELBO = -E_q log p(x|z) + KL(q(z|x) || p(z))
```

**两个KL不同**：与先验比较的KL出现在可计算训练目标；与真实后验比较的KL是下界差距，通常不能直接算。不要说ELBO与似然的差距就是训练目标的先验KL。

## 3. 重建项与KL的分工

重建负对数似然鼓励z能解释x；先验KL限制每张图的编码分布偏离先验。最大化ELBO等价最小化负ELBO，因此目标是重建NLL加正KL，不能写成减KL。

不是把所有样本编码成同一个固定向量，也不强制每一步mu=0、sigma=1。两项共同优化，完全忽略任一项会改变学习目标。beta加权KL是变体，beta不为1时不能直接当作原始负ELBO恒等式。

## 4. 高斯KL为什么有这个形式？

一维q=N(mu,sigma²)，先验p=N(0,1)，两个高斯对数密度相减再取q下期望。利用E_q[(z−mu)²]=sigma²、E_q[z²]=mu²+sigma²得到：

```text
KL = 0.5 * (mu² + sigma² - 1 - log(sigma²))
```

对角多维高斯将各维项相加。mu=1、sigma=2时，KL=.5*(1+4−1−ln4)=1.306853；mu=0、sigma=1时KL=0。

编码器logvar=log(sigma²)，所以var=exp(logvar)、std=exp(0.5*logvar)：

```python
kl_per_sample = 0.5 * (mu**2 + np.exp(logvar) - 1 - logvar).sum(axis=-1)
```

按潜变量维求和，再按batch平均。重建项若按特征维求和，不能随意改成每像素平均而不考虑相对尺度。

## 5. 重建NLL为什么有时是平方误差？

若p(x|z)=N(decoder_mean,固定sigma_x²)，每维NLL=.5*(x−mean)²/sigma_x²+.5*log(2*pi*sigma_x²)。固定方差时与平方误差成比例，常数不影响参数梯度，但计算真实概率值不能随便丢常数。

若使用Bernoulli观测模型，则出现二元交叉熵；观测分布与数据定义要匹配。VAE不是只能用MSE，也不能把潜变量sigma与观测噪声sigma_x混为一谈。

## 6. 重参数化把随机源与参数分开

```python
epsilon = rng.normal(size=mu.shape)
std = np.exp(0.5 * logvar)
z = mu + std * epsilon
```

mu=1、std=2、epsilon=.5时z=2。随机源epsilon独立于参数，但z通过普通运算依赖编码器输出：dz/dmu=1，dz/dlogvar=.5*std*epsilon。

重参数化没有消除随机性。数值梯度检查时必须固定同一epsilon，避免将随机变化误认为参数变化；实际随机训练不应把固定噪声单步实验当作完整期望目标训练。

## 7. 完整标量教学模型与反传

输入标量x，编码mu=a*x+b、logvar=c*x+d；解码均值=w*z+e，观测方差1。每batch样本独立抽一个epsilon，损失取batch平均：

```text
L = mean(0.5*((decoder_mean-x)² + log(2*pi))
         + 0.5*(mu² + exp(logvar) - 1 - logvar))
```

令N为batch数，dmean=(decoder_mean−x)/N，dz=dmean*w，则：

```text
dmu = dz + mu/N
dlogvar = dz*0.5*std*epsilon + 0.5*(exp(logvar)-1)/N
```

各自第一项来自重建路径，第二项来自KL路径。再传给线性编码器和解码器参数，梯度为对应输入与上游梯度乘积汇总。代码同时核对输入梯度；因为x也作为重建目标，需包含目标路径负号。

## 8. 精确下界验证与随机估计的区别

本例线性高斯可以解析积分：z~N(0,1)，x|z~N(w*z+e,1)，因此x~N(e,1+w²)，真实后验方差1/(1+w²)，均值w*(x−e)/(1+w²)。

指定x=.7、近似mu=.2、var=.8、w=.8、e=−.1：精确ELBO=−1.411310，log p(x)=−1.361409，差距=.049902，与后验KL一致。将q改成真实后验，差距为0。

ELBO下界针对完整期望；单次Monte Carlo的随机估计可因抽样波动高于真实log p(x)，不能逐次断言每个采样数都是严格下界。代码分别计算精确期望验证和固定噪声梯度，避免混淆。

## 9. 生成与后验坍塌

训练后生成从先验抽z，不需要编码器，解码器给观测分布，再抽x或展示均值。常见后验坍塌是解码器忽略z，q趋于先验、KL很小；小KL不单独证明训练成功，也不等于对比表示所有样本输出同一个向量。

本标量模型w=0时解码器与z无关，真实后验就是先验，代码核对这一性质。这是现象解释，不是声称已经在真实VAE训练中测到坍塌。

## 10. 核验与来源

[lecture13_vae.py](../experiments/deep_review/lecture13_vae.py)运行完整标量前后向，参数和输入最大绝对差分误差约1.77×10⁻¹¹、1.38×10⁻¹¹。固定噪声下单步损失1.135684→1.132294；另验证Gaussian KL、精确ELBO差距、真后验等号及先验生成路径。没有图像VAE或数据集训练、没有生成质量成绩。

来源：[官方第十三讲课件](https://cs231n.stanford.edu/slides/2025/lecture_13.pdf)、[视频P13](https://www.bilibili.com/video/BV1aXhJ64EmW/?p=13)。

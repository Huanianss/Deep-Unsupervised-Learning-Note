本文学习翻译自哈佛NLP组的博文：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) [<sup>[1]</sup>](#refer-anchor)

原论文：

<img src="fig/fig1.png" width="70%">

​	过去[Attention is All You Need](https://arxiv.org/abs/1706.03762)[<sup>[2]</sup>](#refer-anchor)这篇文章令人印象非常深刻。除了在翻译质量上有较大提升外，它也为其他NLP任务提供了新的神经网络结构。原论文写的非常清楚，但是在编程实现上往往有困难。

​	在这篇博文中，将对论文的实现进行**逐行注释**。已经重排并且删除了一些原始文章内容，并且都添加了注解。此文在Notebook环境下是完全可运行的。全文包括400行代码，在GPU上每秒可处理2.7万个字符(token)。

​	本文代码基础[Pytorch](http://pytorch.org/)框架。完整的notebook已经在[github](https://github.com/harvardnlp/annotated-transformer)[<sup>[3]</sup>](#refer-anchor)上开源，并且可执行在谷歌[Colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing)[<sup>[4]</sup>](#refer-anchor)上。

​	对于研究者与感兴趣的开发者，这仅仅是第一步。代码已经集成在[OpenNMT](https://opennmt.net/)[<sup>[5]</sup>](#refer-anchor)工具包中。(如果有帮助请引用)。对于其他框架的完整实现，可参考 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)[<sup>[6]</sup>](#refer-anchor)(tensorflow) 和[Sockeye](https://github.com/awslabs/sockeye)[<sup>[7]</sup>](#refer-anchor)(mxnet)

- 作者Alexander Rush ([@harvardnlp](https://twitter.com/harvardnlp) or srush@seas.harvard.edu)

### 1 目录

- 依赖库
- 背景
- 模型结构
  - 编码器与解码器堆叠
    - 编码器
    - 解码器
    - 注意力
    - 模型中注意力的应用
  - 位置层面的前馈神经网络
  - 嵌入与软最大
  - 位置编码
  - 完整模型
- 训练
  - 批次和掩码
  - 训练循环
  - 训练数据与批次
  - 硬件与清单
  - 优化器
  - 正则化
    - 标签平滑
- 第一个例子
  - 合成数据
  - 损失计算
  - 贪婪解码
- 现实例子
  - 数据加载
  - 迭代
  - 多GPU训练
  - 训练系统
- 额外的模块：BPE，搜索，平均
- 结果
  - 注意力可视化
- 结论

> 本博文正文主要来自原始论文，注释在这种块内部。

### 2 依赖库

程序的一些依赖库

```Python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

### 3 背景

​	以降低序列计算量为目标构成了扩展神经GPU(Extended Neural GPU)，ByteNet 和ConvS2S的基础。这些都以卷积神经网络为基础模块，并行计算所有输入输出位置的隐藏表示。在这些模型中，将来自两个任意输入或输出位置的信号联系起来所需的操作次数随着位置之间的距离而增长，对于ConvS2S来说是线性的，对于ByteNet来说是对数的。这使得学习遥远位置之间的依赖关系变得很困难。在Transformer中，这种情况被减少到了恒定的操作次数，尽管代价是由于注意力加权位置的平均化而降低了有效的分辨率，我们用多头注意力来抵消这种效果。

​	自注意力，有时也被称为内注意，是一种将单个序列的不同位置联系起来以计算序列表示的注意力机制。自我注意已经成功地应用于各种任务中，包括阅读理解、抽象总结、文本内涵和学习任务无关的句子表示。端到端记忆网络是基于循环注意力机制而不是序列对齐的循环，并且已经被证明在简单语言问答和语言建模任务中表现良好。

​	然而，据我们所知，Transformer是第一个完全依靠自我注意力来计算其输入和输出表示的编码-解码模型，而不使用序列对齐的RNNs或CNNs。

### 4 模型结构

大部分有竞争力的序列转换模型都是编码-解码结构[<sup>[8]</sup>](#refer-anchor)。其中编码器将输入序列的符号表示($x_1,\dots,x_n$)映射为连续的序列表示$\boldsymbol{z}=(z_1,\dots,z_n)$。给定$\boldsymbol{z}$后，解码器将生成输出序列的表示($y_1,\dots,y_m$)。在每一步中，模型是自回归的[<sup>[9]</sup>](#refer-anchor)，即将之前的输出作为预测当前输出的条件。

- 标准编解码模型代码

```Python
class EncoderDecoder(nn.Module):
    """
    标准的编码-解码模型代码。自注意力以及一些其他模型的基础。
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "处理带有掩码的源序列与目标序列"
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

```Python
class Generator(nn.Module):
    "定义标准的 linear + softmax 生成步."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

Transformer以此为基础使用自注意力与点乘堆叠，编码器与解码器都是全连接层，模型结构如下图所示。

<img src="fig/fig2.png" width="60%">

### 4.1 编码器与解码器的堆叠层

#### 4.1.1 编码器 

模型结构图的左半部分由$N=6$个相同层叠加而成。

```python
def clones(module, N):
    "复制多个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

```python
class Encoder(nn.Module):
    "编码器是N个相同层的叠加"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "输入与掩码依次通过每个层"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

网络中在每两个子层间使用残差连接[<sup>[10]</sup>](#refer-anchor)，然后接层归一化(Layer normalization[<sup>[11]</sup>](#refer-anchor)):

```python
class LayerNorm(nn.Module):
    "搭建一个层归一化模块，(具体可阅读参考文献)"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

每个子层的输出是$\text{LayerNorm}(x+\text{Sublayer}(x))$，其中$\text{Sublayer}(x)$是子层本身代表的函数。在子层输出与输入相加前，每个子层的输出都使用dropout[<sup>[12]</sup>](#refer-anchor) 。

为了便于实现残差连接，每个模型的子层维度都是$d_{model}=512$。

```python
class SublayerConnection(nn.Module):
    """
    残差连接后接层归一化。
    为了简化代码层归一化作用在输入上。
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "对于相同尺寸的子层使用残差连接。"
        return x + self.dropout(sublayer(self.norm(x)))
```

每层都有两个子层。第一个是多头自注意力机制，第二个是一个简单的含有点乘的全连接前馈神经网络。

```python
class EncoderLayer(nn.Module):
    "编码器由自注意力层与前馈层组成"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "与网络结构图的左半部分对应"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

#### 4.1.2 解码器 

解码器也是由$N=6$层相同的层组成。

```python
class Decoder(nn.Module):
    "带有掩码的解码器"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

在上文每个编码器层中，包含2个子层。而此处的解码器层包含有3个子层。在中间多添加了一层编码器输出与解码器中间变量的多头注意力。与编码器相似，解码器也在子层中使用残差连接，后接层归一化。

```python
class DecoderLayer(nn.Module):
    "解码器由三部分组成 self-attn, src-attn, and feed forward "
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

解码器中的第一个自注意力层添加了掩码。这种掩码是输出向量序列偏移了一个位置，这保证了预测第$i$个位置时与当前输出之前的输出值有关(注释：例如在翻译任务中，预测输出句子中的单词时，是一个词一个词的预测，且在预测当前词汇时，可以结合已经预测值做出估计。)。

```python
def subsequent_mask(size):
    "屏蔽后续位置"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')#上三角
    return torch.from_numpy(subsequent_mask) == 0
```

> 下图注意力的掩码展示了预测当前词(行)时能够看到的位置(列)。在训练过程中，后面的词汇被屏蔽掉。

```python
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```

<img src="fig/fig3.png" width="60%">

#### 4.1.3 注意力

​	注意力函数可以被描述为这样一种映射，将查询(query)和一组键值对(key-value)映射到输出。其中查询($q$)、键值($k$)、值($v$)和输出都是向量。输出是值v的加权和，权重由查询$q$与对应键值$k$的函数值给出。

​	这种注意力被称为“放缩点乘注意力”。输入由$d_k$维查询$q$与键值$k$和$d_v$维值$v$构成。**注意力计算方法**：计算$q$与所有$k$的点乘，除以$\sqrt{{d_k}}$，再通过Softmax函数得到$v$上的权重。图示如下：

<img src="fig/fig4.png" width="40%">

在代码实现中，可以同时计算很多查询$q$的注意力函数，将所有查询放入矩阵$Q$中。$k$, $ v$也放入矩阵$K$,$V$中。注意力层输出结果如下：

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{{d_k}}})V$$

```python
def attention(query, key, value, mask=None, dropout=None):
    "计算'放缩点乘注意力'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

​	两个最常用的注意力函数是加和注意力[<sup>[13]</sup>](#refer-anchor)与点乘注意力。除了缩放因子$\frac{1}{\sqrt{{d_k}}}$，点乘注意力与我们算法相同。加性注意力计算使用单隐层前馈神经网络计算。虽然二者在理论复杂度方面相似，但是点乘注意力在实践中更快、空间效率更高，因为它可以使用高度优化的矩阵乘积代码。

​	当$d_k$较小时，加和注意力与点乘注意力(不含有放缩因子)性能相近，但是对于较大的$d_k$，加和注意力性能更好[<sup>[14]</sup>](#refer-anchor) 。我们怀疑对于较大的$d_k$，点乘的幅值会迅速变大，使得结果迅速进入Softmax的低梯度区域。(为了解释点乘结果变大，假设$q$, $k$是相互独立的随机变量，0均值，1方差。则它们点乘为$q\cdot k=\sum_{i=1}^{{d_k}}q_ik_i$,其均值为0，方差为$d_k$ )。为了消除方差随着维度变大的影响，我们使用$\frac{1}{\sqrt{d_k}}$进行归一化。

<img src="fig/fig5.png" width="40%">

**多头注意力**：多头注意力允许同时关注在不同位置不同表征子空间的信息。而在单头注意力下，平均化会抑制这一点。

$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^{O}$

其中$\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$

以上投影矩阵分别为：$W_i^Q\in\mathbb{R}^{d_{model}\times d_k},W_i^K\in\mathbb{R}^{d_{model}\times d_k},W_i^V\in\mathbb{R}^{d_{model}\times d_v},W^O\in \mathbb{R}^{hd_v\times d_{model}}$

在原文中，使用$h=8$个并行的注意力层，即8个头。其他维度设置为:$d_k=d_v=d_{model}/h=64$.因为降低了每个头的维度，总的计算量与原始维度单头情况下接近。

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "传入模型维度与头的数目"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设 d_v = d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "计算上图所示多头注意力"
        if mask is not None:
            # h个头上使用相同的掩码.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) 在一批数据中,将q,k,v传入linear层,分裂结果产生多个头的q,k,v
        #    维度变化 d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) 在一批数据中对所有向量进行注意力计算. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" 改变结果形状,并且通过最后一个linear层. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```



#### 4.1.4 注意力在模型中的应用

Transformer以三种方式使用多头注意力：

- 1) 在“编码-解码注意力”层中，查询$q$来自于之前的解码器层，并且缓存的键值$k$与值$v$来自于编码器的输出。这使得解码器可以关注输出序列中每个位置与输入序列所有位置的关联。 这模仿了序列到序列(sequence-to-sequence)模型中典型的编码器-解码器注意机制，如[<sup>[15]</sup>](#refer-anchor)。

- 2) 编码器包括自注意力层。在自注意力层中所有的$k,q,v$来自于上一层编码器的输出。编码器的每个位置都可以关注前一层所有位置的值(*而CNN层间是局部连接*)。
- 3) 相似地，解码器中含有掩码的自注意力层可以关注自身位置之前的输出。阻止信息向左流动，保证自回归性质(*例如：在预测序列 $x_1,x_2,x_3$,...时,拟合的是概率分布$p(x_1),p(x_2|x_1),p(x_3|x_1,x_2)$，序列是逐词预测* )。在代码中使用上述掩码实现这一点。



### 4.2 位置层面的前馈神经网络

除了注意力子层，编解码器的每层中都含有全连接前馈神经网络(FFN)，分开独立地在每个位置上执行。它包括两个线性层，层间使用ReLU激活函数。

$\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}$

虽然不同位置的线性变换相同，但是层的参数不一样。实际上执行的是卷积核大小为1的卷积运算(*此处层间是局部连接：输入位置x到输出位置x之间全连接，不同位置不连接*)。输入输出的维度为$d_{model}=512$，隐藏层维度$d_{ff}=2048$。

```python
class PositionwiseFeedForward(nn.Module):
    "实现FFN."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### 4.3 词嵌入与Softmax

与其他序列转换模型相似，我们使用学习完成的词嵌入(embedding)将输入输出语句转换为$d_{model}$维向量。我们也使用学习完成的线性层和Softmax将解码器的输出转化为预测下一个词的概率。在模型中，在我们的模型中，我们在两个词嵌入层和预Softmax线性变换之间共享相同的权重矩阵，类似于[<sup>[16]</sup>](#refer-anchor) 。在嵌入层中，我们将值乘以$\sqrt{d_{model}}$。

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

### 4.4 位置编码

因为模型中没有包含循环以及卷积操作，所以模型无法利用输入序列的顺序信息。所以在模型语句上加入了表示绝对位置或者相对位置的编码。具体是在编码器与解码器的输入位置加入了“位置编码”。位置编码与词向量具有相同的维度$d_{model}$，所以二者可以相加。有很多的位置编码，有可学习的、有固定的[<sup>[17]</sup>](#refer-anchor) 。

​	在本文中，使用的是不同频率的正余弦函数作为位置编码：	

- $PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$
- $PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$

$pos$是位置，$i$是维度。对于某个固定维度，位置编码是一条正弦曲线。波长范围是$2\pi$到$10000\cdot2\pi$。这种编码可以使得模型轻易学习到序列中的相对位置关系，因为对于任意固定偏置$k$，$PE_{pos+k}$可以表示为$PE_{pos}$的线性函数。

​	另外在编码器与解码器中，位置编码与序列相加后使用dropout。在代码实现中选取$P_{drop}=0.1$

```python
class PositionalEncoding(nn.Module):
    "计算位置编码."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 在对数空间计算位置编码.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
```

> 以下的位置编码在每个维度上频率与偏置都不一样

```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
```

<img src="fig/fig6.png" width="80%">

​	论文作者还试验了使用可学习的位置编码[<sup>[18]</sup>](#refer-anchor)，并发现这两个版本产生的结果几乎是相同的。我们选择了正弦版本，因为它可能允许模型外推到比训练过程中遇到的序列长度更长的序列。

### 4.5 完整模型

> 以下代码有一些超参数用以产生完整的模型

```python
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "提示: 结合一些超参数来构建模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # 由原作者代码导入. 
    # 初始化模型参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```

```python
# 小模型
tmp_model = make_model(10, 10, 2)
None
```



### 5 训练

这部分描述模型的训练。

> 首先需要构造训练数据与掩码

### 5.1 批次与掩码

```python
class Batch:
    "在训练过程中，生成一批带mask的数据。"
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个mask屏蔽padding和当前位置之后的词汇"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

> 接下来我们创建一个通用的训练和评分函数来追踪损失。模型参数的更新也将纳入此函数。

### 5.2 训练循环

``` python
def run_epoch(data_iter, model, loss_compute):
    "标准训练与记录日志函数"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```

### 5.3 训练数据与批次

我们在标准的WMT 2014英德数据集上进行了训练，该数据集由大约450万句子对组成。句子使用字节对编码，其共享的源-目标词汇约为37000个tokens。对于英语-法语，我们使用了明显更大的WMT 2014英法数据集，该数据集由3千6百万个句子组成，并将tokens拆分为32000个词片词汇。

句子对被按近似序列长度分批在一起。每个训练批次都包含一组句子对，包含大约25000个源标记和25000个目标标记。

> 我们将使用torchtext进行批处理。这将在下面详细讨论。在这里，我们在torchtext函数中创建批处理，以确保我们的批处理大小填充到最大的批处理大小不超过一个阈值（25000，如果我们有8个GPU）。

```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

### 5.4 硬件与清单

我们在一台拥有8个NVIDIA P100 GPU的机器上训练我们的模型。对于我们的基础模型，使用本文中描述的超参数，每个训练步骤大约需要0.4秒。我们总共训练了100,000步或12小时的基础模型。对于我们的大模型，步长为1.0秒。大模型的训练时间为30万步（3.5天）。

### 5.5 优化器

我们使用Adam优化器[<sup>[19]</sup>](#refer-anchor)，$\beta_1=0.9$，$\beta_2=0.98$，$\epsilon=10^{-9}$。我们根据公式，在训练过程中改变学习率。这相当于在前几步$warmup\_steps$训练步数中线性增加学习率，之后按步数的平方根反比减少学习率。我们使用$warmup\_steps=4000$。

> 注意：这部分很重要。在模型训练中，需要与这种热启动结合。

``` python
class NoamOpt:
    "封装优化器(改变学习率)."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "更新参数与学习率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "执行学习率更新"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

> 不同模型大小和超参数时，学习率随迭代步长变化曲线。

``` python
# 不同模型超参数下，学习率曲线
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```

![img](.\fig\fig7.png)

### 5.6 正则化

#### 5.6.1 标签平滑

在训练过程中，采用了$\epsilon_{ls}=0.1$对标签进行平滑[<sup>[20]</sup>](#refer-anchor)。折让模型学习到了更多的不确定性，但是准确率与BLEU得分却上升了。

> 文中使用KL散度损失实现标签平滑。文中创建了一个分布对正确词汇有高置信度，对剩余词汇概率是平滑的，而不是使用one-hot目标分布，

```python
class LabelSmoothing(nn.Module):
    "执行标签平滑"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

> 下面一个例子展示，词汇的概率质量基于置信度时如何分布的。

```python
# 标签平滑例子
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))

# 展示系统期望的目标分布
plt.imshow(crit.true_dist)
None
```

![img](.\fig\fig8.png)

> 下面一个例子展示，如果预测结果对于某一类置信度很高时，标签平滑是如何惩罚(约束)这种情况的。

```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```

![img](.\fig\fig9.png)

### 6 第一个例子

> 从一个简单的复制任务开始。从一个小的词汇表中随机给一些输入字符，目标是生成这些字符。

```python
def data_gen(V, batch, nbatches):
    "为源-目标复制任务生成一些数据"
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
```

#### 6.2 损失计算

```python
class SimpleLossCompute:
    "一个简单的损失计算与训练函数"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward() # 注意此处可能会在验证集或测试集上积累梯度
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
```

#### 6.3 贪婪解码

```python
# 训练一个简单的复制任务
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))
```

```
训练过程

Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
1.9274832487106324
Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
1.657595729827881
Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
1.4888023376464843
Epoch Step: 1 Loss: 1.682198 Tokens per Sec: 432.092305
Epoch Step: 1 Loss: 1.313169 Tokens per Sec: 639.441857
1.3485562801361084
Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
0.9853351473808288
Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
0.5686767101287842
Epoch Step: 1 Loss: 0.997076 Tokens per Sec: 433.009746
Epoch Step: 1 Loss: 0.343118 Tokens per Sec: 642.288427
0.34273059368133546
Epoch Step: 1 Loss: 0.459483 Tokens per Sec: 434.594030
Epoch Step: 1 Loss: 0.290385 Tokens per Sec: 642.519464
0.2612409472465515
Epoch Step: 1 Loss: 1.031042 Tokens per Sec: 434.557008
Epoch Step: 1 Loss: 0.437069 Tokens per Sec: 643.630322
0.4323212027549744
Epoch Step: 1 Loss: 0.617165 Tokens per Sec: 436.652626
Epoch Step: 1 Loss: 0.258793 Tokens per Sec: 644.372296
0.27331129014492034
```

> 简单起见，以下代码使用贪婪解码预测翻译结果。

```python 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

` 1     2     3     4     5     6     7     8     9    10
[torch.LongTensor of size 1x10]`



### 7 现实例子

> 现在我们考虑使用IWSLT德英翻译任务的一个实例。这个任务比论文中考虑的WMT任务小得多，但它可以说明整个系统的运作。并且展示如何使用多GPU处理来使其加速运行。

```
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
```

#### 7.1 数据加载

> 使用`torchtext`和`spacy`加载数据，并且按照token (语义级别的词汇切分)分割开。

```python
# 数据加载.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```

> 批次对学习速度影响很大。我们希望有非常均匀的批次，总体使用最少的padding。要做到这一点，我们必须对默认的 `torchtext` 批次进行一些改进。这段代码修补了他们的默认批处理，以确保我们搜索到的句子足够多，从而找到紧凑的批处理。

#### 7.2 迭代器

```python
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "修改torchtext的顺序以匹配我们的模型"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
```



#### 7.3 多GPU训练

> 最后针对快速训练，我们将使用多GPU。这段代码实现了多GPU词生成。并不只是针对transformer的，所以这里并不详细介绍。我们的想法是在训练时将单词生成分割成块，在许多不同的GPU上并行处理。我们使用 pytorch 并行库来实现这一目的。

- replicate - 将模块分到不同GPU上；
- scatter - 将批数据分到不同GPU上；
- parallel_apply -将模块应用到不同的GPU上的批数据；
- gather - 将分散的数据拉回到一个GPU上；
- nn.DataParallel - 一个特殊的模块包，在评估前调用。

```python
# 如果对多GPU不感兴趣可跳过.
class MultiGPULossCompute:
    "多GPU损失计算与训练函数."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # 发送到不同的GPU.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # 将生成结果分块.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # 预测分布
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算损失. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # 求和并归一化损失
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # 对transformer的输出进行反向传播
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # 在transformer中反向传播所有损失            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```

> 现在我们创建我们的模型，评价标准，优化器，数据迭代器与并行。

```python
# 使用的GPU
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
None
```

> 现在我们训练我们的模型。将使用warmup开始训练，其他一切按照默认参数。在AWS的p.8xlarge上使用4个Tesla V100进行训练，每秒运算27000个token，批数据大小为12000。

#### 7.4 训练系统

``` 
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
```

```python
if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
```

> 一旦训练好了，我们就可以对模型进行解码，产生一组翻译。这里我们只需翻译验证集中的第一句话。这个数据集相当小，所以用贪婪搜索的翻译是相当准确的。

```python
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
```

`Translation:	<unk> <unk> . In my language , that means , thank you very much . 
Gold:	<unk> <unk> . It means in my language , thank you very much .  `



### 8 额外的模块：BPE，搜索，平均

> 所以这主要是涵盖了变压器模型本身。有四个方面我们没有明确地覆盖。我们还在OpenNMT-py中实现了所有这些附加功能。

> 1）BPE/字片。我们可以使用一个库先把数据预处理成子字单元。参见Rico Sennrich的subword- nmt实现。这些模型会将训练数据转化成这样的样子。

▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

> 2）共享嵌入。当使用共享词汇的BPE时，我们可以在源/目标/生成器之间共享相同的权重向量。详情请参见[<sup>[21]</sup>](#refer-anchor) 。要将此添加到模型中，只需这样做。

```python
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
```

> 3）光束搜索。这个有点太复杂了，这里就不多说了。参见OpenNMT- py中的pytorch实现。

> 4）模型平均。本文对最后的k个检查点进行平均，以产生集合效应。如果我们有一堆模型，我们可以在事后这样做：

```python
def average(model, models):
    "平均模型"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
```



### 9 结果

在WMT 2014英译德任务上，Transformer（big）（在表2中）比之前报道的SOTA（包括集成模型）的表现要好2.0 BLEU以上，建立了新的SOTA  BLEU得分28.4。该模型的配置列在表3的底行。训练在8个P100 GPU上花了3.5天。即使我们的基础模型也超越了所有之前发布的模型和集成模型，而训练成本只是它们的一小部分。

在WMT 2014英译法任务上，我们的大模型实现了41.0的BLEU得分，超越了之前发布的所有单体模型，而训练成本不到之前SOTA模型的1/4。为英译法训练的Transformer（大）模型使用的dropout $P_{drop}=0.1$，而不是$0.3$。

``` python
Image(filename="images/results.png")
```

![img](.\fig\fig10.png)

> 我们在这里写的代码是基础模型的一个版本。这里有这个系统的完全训练版本（示例模型）。
>
> 通过上一节中的附加扩展，OpenNMT-py复制在EN-DE WMT上达到了26.9。在这里，我已经把这些参数加载到我们的重新实现中。

```python
!wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
```

```python
model, SRC, TGT = torch.load("en-de-model.pt")
```

```python
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
```

`Translation:	<s> ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden . `

#### 9.1注意力可视化

> 即使是用贪婪的解码器，翻译出来的效果也很不错。我们可以进一步将其可视化，看看在注意力的每一层发生了什么。

```python
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()		
```



`编码器层 2`

![img](.\fig\enc_l2.png)



`编码器层 4`

![img](.\fig\enc_l4.png)



`编码器层 6`

![img](.\fig\enc_l6.png)



`解码器 Self 层2`

![img](.\fig\dec_self_l2.png)



`解码器 Src 层2`

![img](.\fig\dec_src_l2.png)



`解码器 Self 层4`

![img](.\fig\dec_self_l4.png)



`解码器 Src 层4`

![img](.\fig\dec_src_l4.png)



`解码器 Self 层6`

![img](.\fig\dec_self_l6.png)



`解码器 Src 层6`

![img](.\fig\dec_src_l6.png)

### 10结论

>  希望这段代码对以后的研究有用。如果你有任何问题，请联系我们。如果你觉得这段代码有帮助，也可以看看我们其他的OpenNMT工具。

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```



<div id="refer-anchor"></div>

#### 参考

[1] http://nlp.seas.harvard.edu/2018/04/03/attention.html

[2] https://arxiv.org/abs/1706.03762

[3] https://github.com/harvardnlp/annotated-transformer

[4] https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing

[5] https://opennmt.net/

[6] https://github.com/tensorflow/tensor2tensor

[7] https://github.com/awslabs/sockeye

[8] https://arxiv.org/abs/1409.0473

[9] https://arxiv.org/abs/1308.0850 

[10] https://arxiv.org/abs/1512.03385

\[11] https://arxiv.org/abs/1607.06450 

[12] http://jmlr.org/papers/v15/srivastava14a.html

[13] https://arxiv.org/abs/1409.0473

[14] https://arxiv.org/abs/1703.03906

[15] https://arxiv.org/abs/1609.08144

[16] https://arxiv.org/abs/1608.05859

[17] https://arxiv.org/pdf/1705.03122.pdf

[18] https://arxiv.org/pdf/1705.03122.pdf

[19] https://arxiv.org/abs/1412.6980

[20] https://arxiv.org/abs/1512.00567

[21] https://arxiv.org/abs/1608.05859
# Automatic-generation-of-text-summaries
使用两种方法（抽取式Textrank和概要式seq2seq）自动提取文本摘要

主流的文本摘要方式

目前主流的文本摘要自动生成有两种方式，一种是抽取式（extractive），另一种是生成式（abstractive）。

抽取式顾名思义，就是按照一定的权重，从原文中寻找跟中心思想最接近的一条或几条句子。而生成式则是计算机通读原文后，在理解整篇文章意思的基础上，按自己的话生成流畅的翻译。

TextRank算法是根据google的pagerank算法改造得来的，google用pagerank算法来计算网页的重要性。textrank在pagerank的原理上用来计算一个句子在整个文章里面的重要性，下面通过一个例子来说明一下:
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/6.png)

图中每个球都代表了一张网页，每一个箭头代表该网页上有其它网页的超链接，如D球，它有指向A的箭头，代表D网页上有A的连接，而E指向了D表示E网页上有D的连接。被引用的越多代表该网页的重要性越大。表格如下<br>

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/13.jpg)

通过这张图可以计算每张网页被其它网页引用的次数，从而算出这张网页的重要程度，计算公式如下：

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/10.png)


式中d代表阻尼系数，d∈[0,1]，一般取d=0.85。对于B来说，有三个页面推荐了B，S(vi)代表的是页面的初始分数，这里一般设置为1，也可以设置成其他。所以S（B）=（1-0.85）+0.85*（（1/2）*1+（1/2）*1+（1/2）*1）便是页面B的分数，依次计算得到所有页面的分数。再将页面分数rank取topN就可以得到N个最重要的页面。
<br>
在textrank中，人们用句子的相似性来取代网页之间的相互链接的个数。<br>

前面提到计算两个网页之间的互相引用的次数从而得打网页的重要性，那么句子之间的连续如何建立了？传统的方法是比较句子中相同单词的个数，比如“I am a dog”,"you are a dog"这两个句子有连个相同的单词"a","dog"。这两个单词同属于两个句子，因此S(si,sj)=2/log(4)+log(4)。<br>

有了句子的相似度，我们就可以建立邻接矩阵。注意到相似度是不分出度和入度的，因此邻接矩阵是关于对角线对称的矩阵<br>

通过相似度矩阵，我们就可以通过以下公式来计算句子的重要程度：

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/7.png)

我们注意到这种传统的句子相似性在某种程度上使句子之间建立起了联系，但是单词的词性，单词的近义词，反义词等诸多因素都未考虑进去，因此这种计算句子之间相似性的方法并不优秀。

基于以上背景，本项目的抽取式概要生成方法是基于word2vec的基础再通过textrank的算法来获得文本摘要。既然传统的计算句子相似性的算法不能够满足现在的要求，那么通过word2vec来计算句子相似性。在word2vec中，每个单词都被表示成了向量，这些向量通过单词之间的联系建立起关系。

本项目使用的计算句子相似度的方式如下：

如计算句子A=['word','you','me']，与句子B=['sentence','google','python']计算相似性，从word2vec模型中分别得到A中三个单词的词向量v1,v2,v3取其平均值Va(avg)=(v1+v2+v3)/3。对句子B做同样的处理得到Vb(avg)，然后计算Va(avg)与Vb(avg)连个向量的夹角余弦值，Cosine Similarity视为句子A与B的相似度<br>


![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/5.jpg)

有了句子的相似度，则可通过以下过程来计算句子的重要程度：<br>

（1）将article分句，装成一个链表。<br>

（2）再将上述的链表中的每一句（sentence）分词，这里推荐jieba分词，当然最好去掉标点符号，以及一些停用词。得到一个二维的list。<br>

（3）然后将每一个句子中的单词，分别于其它句子进行两两相似性计算。<br>

（4）然后计算每一句的相对于另一句的分数，具体方式在上述的pagerank的算法中。<br>

（5）迭代计算每一句的分数，重复迭代，直到分数的差值在0.0001下。<br>

（6）排序上述得到的句子，取分数最高的topN。便是想要得到的句子。<br>

项目的demo地址：textsummary.herokuapp.com<br>

具体的实现包含在代码文件中。

另一种生成式文本摘要提取主要依靠深度神经网络结构实现，2014年由GoogleBrain团队提出的Sequence-to-Sequence序列，开启了NLP中端到端网络的火热研究。Sequence-to-Sequence又称为编、解码器（Encoder、Decoder）架构。其中Encoder、Decoder均由数层RNN／LSTM构成，Encoder负责把原文编码为一个向量C；Decoder负责从这个向量C中提取信息，获取语义，生成文本摘要。

Bahdanau等人在14年发表的论文《Neural Machine Translation by JointlyLearning to Align and Translate》中，第一次将Attention机制应用于NLP中。Attention机制是一种注意力（资源）分配机制，在某个特定时刻，总是重点关注跟它相关的内容，其他内容则进行选择性忽视。就像下图，在翻译“Knowledge”时，只会关注“知识”.这样的对齐能让文本翻译或者摘要生成更具针对性。

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/3.jpg)

传统的seq2seq结构如下图所示：
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/15.jpg)

即通过Encoder将输入语句进行编码得到固定长度的Context Vector向量，这个编码过程实际上是一个信息有损压缩的过程；随后再将Context Vector传给Decoder进行翻译结果的生成，在Decoder端生成每个单词时，均参考来自Encoder端相同的Context Vector

这种方式相对不够灵活，具体而言，当我们在翻译“机器学习”这的词的时候，并不关心这个词组前面的“我”和“爱”这两个字；而在翻译“我”的时候，也不关心“机器学习”这个词组。因此，一种更好的方式就是引入Attention机制，给予当前待翻译的词更多的权重，使得我们翻译每个词时会对源语句有不同的侧重，如下图所示：
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/14.jpg)

上图中不同颜色代表着不同的Context Vector，我们在翻译每个单词时都有不同的Context Vector。以“machine”对应的Context Vector为例，其连接“机器”这个词的线更粗，代表着翻译时给予“机器”这个词更多的Attention，而翻译“learning”时则给予“学习”这个词更多的Attention。

BiRNN正如其名字所说——双向RNN，意思是我们不仅考虑句子的正向序列，还要考虑反向序列，以此让模型捕获句子的完整信息。更直观一点，想象我们在翻译一个句子时，一般会先读完整个句子，观察句子中的前后关系，然后再开始翻译。BiRNN就起到了对整个句子正向序列和反向序列进行观察的作用。

具体的理论内容不再赘述，下面简单介绍下本项目使用的attention机制：<br>

Attention层的核心在于对Encoder端隐层的权重计算。如下图所示：
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/18.jpg)

以“我爱机器学习”为例，假如我们当前时刻准备生成“machine”这个词，此时我们需要计算Context Vector。<br>

我们对图中的符号先进行定义：<br>

1.s_prev代表Decoder端前一轮的隐层状态，即代表了翻译“love”阶段的输出隐层状态；<br>

2.蓝色框图中的a1-a4分别代表了Encoder端每个输入词BiRNN隐层状态。例如，a1代表了“我”这个词经过Bi-LSTM后的输出向量；<br>

3.红色α1-α4分别代表了Attention机制学习到的权重。例如α3代表了“机器”这个词的权重，可以看到α3对应的线条比较粗，意味着在翻译生成“machine”这个词时对应的Context Vector会给予“机器”这个词更多的Attention；<br>

4.绿色的圈圈代表加权后的Context Vector。Attention中权重α是关于a和s_prev的函数，我们首先将Bi-LSTM的隐层状态a和s_prev进行concat；然后利用全连接层并采用Softmax激活函数训练一个小的神经网络，得到输出α。进而再利用得到的权重α对Bi-LSTM的隐层状态a进行加权求和，得到当前翻译“machine”的Context Vector<br>

最后将这个Context Vector输入给Decoder进行处理。对翻译的每个词我们都可以采用同样的方式进行构造。<br>

其他细节不再赘述，代码包含在源文件中。<br>

本项目使用基于Tensorflow的seq2seq+attention框架搭建摘要生成网络。<br>

实验结果分析：
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/9.png)

使用的训练语料是搜狐新闻数据集，包含了40万个新闻-标题对儿；测试数据使用两种，一种是从源数据集里随机选了2000个出来（没有用在训练），另一种是评测数据，包含500个。模型的调参是在验证集上用Rouge（即召回率）这个指标，用Rouge1和Rouge2两个指标对比了三种模型（传统Textrank,抽取式模型，生成式模型），结果表明后两种模型表现更好，Rouge值有显著提高。


生成式模型受训练语料与训练时间的影响较大，相比抽取式模型有不稳定性，但生成式模型产生的结果更简洁，具有总结归纳的能力，是抽取式模型不具备的。


训练数据庞大，学校网速有限，不再上传。有需要联系513617866@qq.com 欢迎指正

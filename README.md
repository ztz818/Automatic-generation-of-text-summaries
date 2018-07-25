# Automatic-generation-of-text-summaries
使用两种方法（抽取式Textrank和概要式seq2seq）自动提取文本摘要

主流的文本摘要方式

目前主流的文本摘要自动生成有两种方式，一种是抽取式（extractive），另一种是生成式（abstractive）。

抽取式顾名思义，就是按照一定的权重，从原文中寻找跟中心思想最接近的一条或几条句子。而生成式则是计算机通读原文后，在理解整篇文章意思的基础上，按自己的话生成流畅的翻译。

TextRank算法是根据google的pagerank算法改造得来的，google用pagerank算法来计算网页的重要性。textrank在pagerank的原理上用来计算一个句子在整个文章里面的重要性，下面通过一个例子来说明一下:
![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/6.png)

图中每个球都代表了一张网页，每一个箭头代表该网页上有其它网页的超链接，如D球，它有指向A的箭头，代表D网页上有A的连接，而E指向了D表示E网页上有D的连接。被引用的越多代表该网页的重要性越大。通过这张图可以计算每张网页被其它网页引用的次数，从而算出这张网页的重要程度，计算公式如下：

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/7.png)


式中d代表阻尼系数，d∈[0,1]，一般取d=0.85。对于B来说，有三个页面推荐了B，S(vi)代表的是页面的初始分数，这里一般设置为1，也可以设置成其他。所以S（B）=（1-0.85）+0.85*（（1/2）*1+（1/2）*1+（1/2）*1）便是页面B的分数，依次计算得到所有页面的分数。再将页面分数rank取topN就可以得到N个最重要的页面。

在textrank中，人们用句子的相似性来取代网页之间的相互链接的个数。公式如下：

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/5.jpg)


前面提到计算两个网页之间的互相引用的次数从而得打网页的重要性，那么句子之间的连续如何建立了？传统的方法是比较句子中相同单词的个数，比如“I am a dog”,"you are a dog"这两个句子有连个相同的单词"a","dog"。这两个单词同属于两个句子，因此S(si,sj)=2/log(4)+log(4)。这种传统的句子相似性在某种程度上使句子之间建立起了联系，但是单词的词性，单词的近义词，反义词等诸多因素都未考虑进去，因此这种计算句子之间相似性的方法并不优秀。但是它却比起之前的词频法和tf*idf的方法有了很大的进步。


本项目的抽取式概要生成方法是基于word2vec的基础再通过textrank的算法来获得文本摘要。既然传统的计算句子相似性的算法不能够满足现在的要求，那么通过word2vec来计算句子相似性。在word2vec中，每个单词都被表示成了向量，这些向量通过单词之间的联系建立起关系。具体的实现包含在代码文件中。

另一种生成式文本摘要提取主要依靠深度神经网络结构实现，2014年由GoogleBrain团队提出的Sequence-to-Sequence序列，开启了NLP中端到端网络的火热研究。Sequence-to-Sequence又称为编、解码器（Encoder、Decoder）架构。其中Encoder、Decoder均由数层RNN／LSTM构成，Encoder负责把原文编码为一个向量C；Decoder负责从这个向量C中提取信息，获取语义，生成文本摘要。

Bahdanau等人在14年发表的论文《Neural Machine Translation by JointlyLearning to Align and Translate》中，第一次将Attention机制应用于NLP中。Attention机制是一种注意力（资源）分配机制，在某个特定时刻，总是重点关注跟它相关的内容，其他内容则进行选择性忽视。就像下图，在翻译“Knowledge”时，只会关注“知识”.这样的对齐能让文本翻译或者摘要生成更具针对性。

![.](https://github.com/ztz818/Automatic-generation-of-text-summaries/blob/master/pictures/3.jpg)

本项目使用基于Tensorflow的seq2seq+attention框架搭建摘要生成网络，具体代码在代码文件中。

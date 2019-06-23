# 1062388964-qq.com
github桌面版

Triplet Network实验代码说明：

一、rename文件：对20_newsgroups进行处理，重命名分类文件，便于下一步处理；

二、handle_chars文件：
（1）将文件整理、归并为一个分类对应一个文件；
（2）预处理文件中的英文数据，包括去停用词、去特殊字符、分词等处理；
（3）整理为每个分类文件包含分好词后的处理结果；

三、reload文件：将分类数据所有文本内容加到all_texts，便于读取；

四、build_vocab文件：将分类好的文本数据处理、生成为vocab.words，即每行是一个id对应一个单词的形式；

五、build_glove文件：根据英文单词对应的Vector词典：glove.840B.300d.txt，生成glove.npz文件，即每个文件中的单词都对应一个Vector；

六、Text2vec文件：针对每个文件，按行读取，得到一个三维的embedding，每个维度分别代表[文本，行，字]，并报存在.pkl文件中

七、lstm_dx文件：Triplet Network主实验程序：
（1）对处理好的embedding数据取一部分训练集样本和一部分测试集样本，并统一reshape为(1000,300,300)的数组格式；
（2）使用孤立森林的方式排除100个异常点，剩下的点作为正常样本；
（3）定义一个两层的LSTM模型
（4）使用batch_hard_triplet_loss或batch_all_triplet_loss损失函数；
（5）使用accuracy、precision、recall等指标对分类结果评测；

八、model文件：Triplet Network底层函数文件，内含batch_hard_triplet_loss、batch_all_triplet_loss、马氏距离度量算法等的原函数。

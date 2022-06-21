# [Kaggle] **U.S. Patent Phrase to Phrase Matching** 银牌方案
### 使用模型：

+ Deberta-v3-large
+ Bert-for-patent
+ Deberta-base
+ Electra

### 尝试使用的处理与技术：

+ 对抗学习FGM模块，增强鲁棒性（部分模型上LB有提升）
+ LR scheduler: 采用Linear和Cosine两种，并调整周期(cosine, 0.5时效果最好)
+ 对context中的单词进行shuffle，以进行数据增强(部分模型上LB有提升)
+ 对context中的单词进行随机mask，进行数据增强(没有效果)
+ 对于bert输出的pooling处理，尝试了self-attention以及Bi-LSTM(部分效果有提升)
+ 预测时对预测值进行分桶(LB有较明显提升)

### 学习前排大佬：
+  Groupby anchor and stratify by score, also there are some words occur in both anchor and target, make sure to put them in the same fold.
+  Random shuffle targets during each training step.
+  Using linear attention pooling on top of BI-LSTM before fc.
+  Change rnn output dim * 2 from (bert out dim like 1024 to 2048) help a lot for some weak models like bert-for-patents and simcse-bert-for-patent.
+  Minmax scale for each model's output before adding to ensemble.
+  ...
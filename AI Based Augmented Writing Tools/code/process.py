#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import MeCab

# 数据预处理
def preprocess(mecab, doc, stopwords):
    doc = doc.strip()
    if len(doc) > 0:
        seg_line = mecab.parse(doc)  # 分词
        # print(seg_line)

        return seg_line


if __name__ == '__main__':
    # 读取停用词表
    # stopwords = [w.strip() for w in open('data/stopwordslist.txt', 'r', encoding='utf-8').readlines()]
    stopwords = []
    #分词第一步，第二部是停用词处理
    #词性筛选 分词之后 根据任务，

    # 读取数据集
    data = pd.read_csv("ldgourmet/ratings.csv")
    data_text = data['body'].tolist()  # 评论数据集
    # print(data_text)

    # mecab = MeCab.Tagger("-Ochasen")  # 带词性标注
    # おわかち
    mecab = MeCab.Tagger("-Owakati")  # 仅仅分词

    # 数据预处理
    corpus = [preprocess(mecab, line, stopwords) for line in data_text if len(line) > 0]
    print(corpus)

    # 输出语料
    with open("result/corpus.text", 'w', encoding="utf-8") as f:
        for text in corpus:
            f.write(text + '\n')

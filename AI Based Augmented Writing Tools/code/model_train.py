# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

if __name__ == '__main__':
    path = "result2/"
    input1 = path + "corpus.text"

    #output1 = path + "corpus.word2vec.model"
    output1 = path + "corpus.doc2vec.model"
    output2 = path + "corpus.vector.model"

    sentences = []
    with open(input1, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if line != '':
                sentences.append(line.split(" "))


    # model = Word2Vec(vector_size=100, window=5, min_count=5, workers=4)  # 定义word2vec对象

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    #model = Doc2Vec(documents, vector_size=2, window=5, min_count=1, workers=4)
    model = Doc2Vec(documents, vector_size=100, window=1, min_count=1, workers=1)

    model.build_vocab(sentences)  # 建立初始训练集的词典
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # 模型训练
    #

    model.save(output1)  # 模型保存
    model.wv.save_word2vec_format(output2, binary=False)  # 词向量保存

'''
# 模型的训练和保存，试了一下下面两行同样可以训练，但是无法追加训练
model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
'''

from __future__ import print_function
from LDA.ulits import *

import jie_test
from gensim import corpora,models

train_set=[]

walk=os.walk(dir_data+datasets)
print(walk)

for root,dir,files in walk:
    for name in files:
        f=open(os.path.join(root,name),
               'r',
               encoding='utf-8',
               errors='ignore')
    raw=f.read()
    word_list=list(jie_test.cut(raw, cut_all=False))
    train_set.append(word_list)
    print(word_list)

dic=corpora.Dictionary(train_set)
print(dic)

corpus=[dic.doc2bow(text) for text in train_set]
tfidf=models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]
print(corpus_tfidf)
LDA=models.LdaModel(corpus_tfidf,
                    id2word=dic,
                    num_topics=10)
corpus_LDA=LDA[corpus_tfidf]


for topic in LDA.print_topics(num_words=5):
    termNumber=topic[0]
    print(topic[0],':',sep='')
    listOfTerms = topic[1].split('+')
    for term in listOfTerms:
        listItems = term.split('*')
        print('  ', listItems[1], '(', listItems[0], ')', sep='')
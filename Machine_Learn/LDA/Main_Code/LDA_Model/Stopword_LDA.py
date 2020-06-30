from __future__ import print_function
import jie_test
from LDA.ulits import dir_data
from LDA.ulits import datasets

def stopwordslist(filepath):
    file=open(filepath,'r',encoding='utf-8')
    lines=file.readlines()
    for line in lines:
        stopwords=line.strip()
        # print(stopwords)
        return stopwords


def seg_sentence(sentence):
    sentence_seged=jie_test.cut(sentence.strip())
    stopwords=stopwordslist(dir_data+'stopwords/stops.txt')
    outstr=''
    for word in sentence_seged:
        if word not in stopwords:
            if word !='\t':
                outstr +=word
                outstr +=" "

    return outstr

inputs=open(dir_data+datasets+'/'+'abstract.txt',
            'r',
            encoding='utf-8',
            errors='ignore')

outputs=open(dir_data+datasets+'/'+'abstract_stops.txt',
             'w',
             encoding='utf-8',
             errors='ignore')


for inputline in inputs:
    inputline_seg=seg_sentence(inputline)
    outputs.write(inputline_seg+'\n')

outputs.close()
inputs.close()







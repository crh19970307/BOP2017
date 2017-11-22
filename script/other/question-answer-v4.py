import numpy as np
import re
import tensorflow as tf
import jieba
import jieba.analyse
import sys
import math
from gensim.models import Word2Vec
import json
# from keras.preprocessing import sequence  
# from keras.optimizers import SGD, RMSprop, Adagrad  
# from keras.utils import np_utils  
# from keras.models import Sequential  
# from keras.layers.core import Dense, Dropout, Activation  
# from keras.layers.embeddings import Embedding  
# from keras.layers.recurrent import LSTM, GRU  
from keras.models import load_model


def spiltword(sentencelist):
    wordlist = []
    # print('\nSplitting word\n')
    for index, storage in enumerate(sentencelist):
        wordsplit = list(jieba.cut(storage, cut_all=False))
        wordlist.append(wordsplit)
    return wordlist


def turnwordtovector(wordlist,model):
    vectorlist = []
    for index, data in enumerate(wordlist):
        tmp = []
        for item in data:
            if item in model.vocab:
                tmp.append(model[item])
        vectorlist.append(tmp)
    return vectorlist


def ifSimilar(vec1, vec2):
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return  0.5+0.5*np.sum(vec1 * vec2) / denom  if denom!=0 else 0# 余弦值  



def loaddata():
    f = open('content.txt', 'r', encoding='utf-8')
    f2=open('baike.txt', 'r', encoding='utf-8')
    r = f.read()
    r2=f2.read()
    r=re.sub(r'|·|●|★|\*|>','',r)
    r2 = re.sub(r'|·|●|★|\*|>', '', r2)
    data = re.sub(r'\s+', ' ', r)
    data2=re.sub(r'\s+', ' ', r2)


    data=re.findall(r'(?<=[,^，\s！？。；;!?])(?=([^,，\s！？。；;!?]+[，,\s！？。；;!?][^,，\s！？。；;!?]*))',data)
    data2=re.findall(r'(?<=[,^，\s！？。；;!?])(?=([^,，\s！？。；;!?]+[，,\s！？。；;!?][^,，\s！？。；;!?]*))',data2)

    model = load_model('0.608113.h5')
    global graph
    graph = tf.get_default_graph()
    #data=[]
    whole_data=data2+data
    ###################
    tmp=[]
    for i,item in enumerate(whole_data):
        if len(re.split('[！？。；;!?]', item)) > 1:
            t=re.split('[！？。；;!?]', item)
            for i in t:
                tmp.append(i)
        else:
            tmp.append(item)
    whole_data=list(set(tmp))
    return whole_data, model
    '''
    f2=open('baike-process.txt','w')
    f2.write(str(data))
    f2.close()
    '''


# print(data[0:10])
def getanswer(question, data, model,vec_model):
    seg_list = jieba.lcut_for_search(question)
    cntlist = []
    ############################
    ####################################
    # 第一步删选
    for index, item in enumerate(data):
        cnt = 0
        for item2 in seg_list:
            if item2 in vec_model.vocab:
                vec1 = vec_model[item2]
                sentence = jieba.lcut_for_search(item)
                for item3 in sentence:
                    if item3 in vec_model.vocab:
                        vec2 = vec_model[item3]
                        sim = ifSimilar(vec1, vec2)
                        if sim > 0.89:#0.9
                            if item2 != item3 and item2 not in  ['是','的','，','？']:
                                cnt = cnt + sim ** 24#30
                                continue
            if item2 in item and item2 not in  ['是','的','，','？']:
                cnt = cnt + 1
        if len(item) != 0:
            #if cnt>0.2:
             #   print(cnt / (len(item) ** 0.25),item)
            cntlist.append(cnt / (len(item) ** 0.25))#0.25
        else:
            cntlist.append(0)
    ###############################


    cntarray = np.array(cntlist)
    indexlist = np.argsort(cntarray)
    indexlist = indexlist.tolist()
    indexlist.reverse()
    sentencelist = []
    answerlist = []
    ##########################
    # 适当切分再删选
    #print(numberofpossibleanswer)
    numberofpossibleanswer=3
    for i in range(numberofpossibleanswer):
        item = data[indexlist[i]]
        if len(item) >= 3:
            sentencelist.append(question + item)
            if (',' in item or '，' in item or ' ' in item) and re.match(r'.*“.+[,，\s].+”.*',item) is None:
                a = re.split('[,，\s]', item)
                for item2 in a:
                    if question + item2 not in sentencelist:
                        words1, words2 = jieba.lcut_for_search(question), jieba.lcut_for_search(item2)
                        for word1 in words1:
                            if word1 in words2 and word1 not in ['是','的','，','？']:
                                sentencelist.append(question + item2)

    sentencelist=list(set(sentencelist))
    #########################
    wordlist = spiltword(sentencelist)
    vectorlist = turnwordtovector(wordlist,vec_model)
    xa = np.zeros((len(vectorlist), 30, 60), dtype='float64')

    for index1, items in enumerate(vectorlist):
        for index2, item2 in enumerate(items):
            if index2 == 30:
                break
            xa[index1][index2] = item2

    global graph
    with graph.as_default():
        result = model.predict(xa)

    tmp = []
    for i in range(len(vectorlist)):
        tmp.append(result[i][0])
    result = np.array(tmp)
    result = np.argsort(result)
    result = result.tolist()
    result.reverse()

    return '\n'.join([sentencelist[i] for i in result[0:1]])
    '''
    for i in range(numberofpossibleanswer):
        print('\n')
        print(tmp[result[i]])
        print(data[indexlist[result[i]]])
    '''


def main():
    data, model = loaddata()
    questionlist = [
        '贵校有本科生多少人？',#1
        '贵校有多少长江学者？',#1
        '贵校的食堂菜怎么样？',#0
        '张杰是贵校的校长吗？',#0
        '贵校毕业生的就业情况怎么样？',#0
        '贵校有多少年的历史？',#0
        '贵校的医学院是和哪所大学合并的？',#0
        '贵校在科研方面有哪些成绩？',  #0
        '贵校在国内的大学中排名如何？',#0
        '贵校的教学水平怎么样？',#0
        '贵校的党委书记是谁？',#1
        '贵校的办学宗旨是什么？',#1
        '贵校有多少名工程院院士？',  # 1
        '贵校坐落于哪座城市/省份？',  # 0
        '贵校的地址在哪里？',#0
        '贵校的校训是什么？',  # 0
        '贵校的校庆是什么时候？',#1
        '贵校的首任校长是谁？',#1
        '贵校有几个国家重点实验室？',  # 1
        '贵校的现任校长是谁？',  # 1
        '贵校有多少学院？',  # 0
        '贵校的官网网址是什么？',#1
        '贵校海外留学的比例有多少？',#1
        '贵校的英文名是什么？',#0
        '贵校的简称是什么？',#1
        '贵校的一级学科有是哪些？',#0
        '贵校的校区面积多大？',#1
        '贵校的校徽图形是什么？',#1
        '贵校的图书馆藏书量是多少？',  # 1
        '贵校建立于哪一年？',  # 0
        '贵校校歌是什么？',  # 0
        '贵校有几个国家一级重点学科？', #1
        '贵校有几个校区？',#1
        '贵校是985大学吗？',#1
        '贵校有多少家附属医院？', #1
        '贵校有哪些知名校友？', #0
        '贵校有几个硕士点？',#1
        '贵校获得了哪些国家级科研奖项？', #0
        '贵校有专职教师多少名？',#1
        '贵校有多少本科专业？',#1


        '贵校总的科研经费是多少？']#0
    vec_model = Word2Vec.load('Word60.model')
    for question in questionlist:
        print(getanswer(question, data, model,vec_model))
        # wordsplit=list(jieba.cut(question,cut_all=False))


if __name__ == '__main__':
    main()

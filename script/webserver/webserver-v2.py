from flask import Flask, jsonify,request
import numpy as np 
import re
import jieba
import jieba.analyse 
import sys
import math
from gensim.models import Word2Vec
import json
import tensorflow as tf 
from keras.models import load_model
def spiltword(sentencelist):
    wordlist = []
    # print('\nSplitting word\n')
    for index, storage in enumerate(sentencelist):
        wordsplit = list(jieba.cut(storage, cut_all=False))
        wordlist.append(wordsplit)
    return wordlist


def turnwordtovector(wordlist):
    # print('\nLoading word2vec model...\n')
    model = Word2Vec.load('Word60.model')
    # print('\nTurning word to vector\n')
    vectorlist = []
    for index, data in enumerate(wordlist):
        tmp = []
        for item in data:
            if item in model.vocab:
                tmp.append(model[item].tolist())
        vectorlist.append(tmp)
    # progressbar(index,len(wordlist))
    return vectorlist

def ifSimilar(vec1,vec2):
    num = float(np.sum(vec1 * vec2))  # 若为行向量则 A * B.T  
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0
    cos = num / denom  # 余弦值  
    sim = 0.5 + 0.5 * cos  # 归一化  
    return sim


def loaddata():
    f = open('baike.txt', 'r', encoding='utf-8')
    r = f.read()
    # print(r[0])
    data = re.sub(r'\s+', ' ', r)
    data = re.split('[！？。；;!?]', data)
    model = load_model('model/0.608113.h5')
    global graph 
    graph = tf.get_default_graph()
    return data, model
    '''
    f2=open('baike-process.txt','w')
    f2.write(str(data))
    f2.close()
    '''


# print(data[0:10])
def getanswer(question, data, model):
    vec_model = Word2Vec.load('Word60.model')
    seg_list = jieba.lcut_for_search(question)
    print(seg_list)
    numberofpossibleanswer = 5
    cntlist = []
    ####################################
    #第一步删选
    for index, item in enumerate(data):
        cnt = 0
        for item2 in seg_list:

            if item2 in vec_model.vocab:
                vec1 = vec_model[item2].tolist()
                sentence = jieba.lcut_for_search(item)
                for item3 in sentence:
                    if item3 in vec_model.vocab:
                        vec2 = vec_model[item3].tolist()
                        sim=ifSimilar(np.array(vec1), np.array(vec2))
                        if sim>0.9:
                            if item2 != item3:
                                cnt = cnt+sim**10
                                continue
            if item2 in item:
                cnt = cnt + 1
        if len(item)!=0:
            cntlist.append(cnt/len(item)**0.3)
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
    #适当切分再删选
    for i in range(numberofpossibleanswer):
        if ',' in data[indexlist[i]] or '，' in data[indexlist[i]]:
            tmp = re.findall(r'(?<=[,\^，\s])(?=([^,，\s]+[，,\s][^,，\s]+))',data[indexlist[i]])
        else:
            tmp=data[indexlist[i]]
        for item in tmp:
            if len(item)>=3:
                sentencelist.append(question + item)
                if ',' in item or '，' in item:
                    a=re.split('[,，]',item)
                    for item2 in a:
                        if question+item2 not in sentencelist:
                            words1,words2=jieba.lcut_for_search(question),jieba.lcut_for_search(item2)
                            for word1 in words1:
                                if word1 in words2:
                                    sentencelist.append(question +item2)
    sentencelist=list(set(sentencelist))
    #########################
    wordlist = spiltword(sentencelist)
    vectorlist = turnwordtovector(wordlist)
    xa = np.zeros((len(vectorlist), 30, 60), dtype='float64')
    for index1, items in enumerate(vectorlist):
        for index2, item2 in enumerate(items):
            if index2 == 30:
                break
            xa[index1][index2] = item2
    global graph
    with graph.as_default():
        result = model.predict(xa)
    # print(result)
    tmp = []
    for i in range(len(vectorlist)):
        tmp.append(result[i][0])
    result = np.array(tmp)
    result = np.argsort(result)
    result = result.tolist()
    result.reverse()
    # print(data[indexlist[result[0]]])
    return '\n'.join([sentencelist[i] for i in result][0:2])
    '''
    for i in range(numberofpossibleanswer):
        print('\n')
        print(tmp[result[i]])
        print(data[indexlist[result[i]]])
    '''

app = Flask(__name__)
'''
@app.route('/', methods=['GET'])
def home():
	return jsonify({'tasks': tasks})
'''

@app.route('/post/', methods=['GET', 'POST'])
def real_time_api():
	if request.method == 'POST':
		#question=request.data
		#question=request.data
		#data2,model2=loaddata()
		#datadict=request.form
		#print(datadict['que'])
		#print(data2)
		#return datadict['que']
		#print('\nrequest.form is :\n')
		#print(request.form)
		# print('\nrequest.args is:\n')
		# print(request.args)
		# print('\nrequest.value is:\n')
		# print(request.values)
		# print('\nrequest.cookies is\n')
		# print(request.cookies)
		# #dataDict = request.data 
		# print('\nrequest.data is :\n')
		# print(request.data)
		dataDict=json.loads(request.data.decode())
		#print(dataDict)
		#print(dataDict)
		#return str(dataDict)
		#return 'hdlkgjahbadks'
		global model2,data2
		#print(model2)
		answer={}
		answer['ans']=getanswer(dataDict['que'],data2,model2)
		print (json.dumps(answer))
		return json.dumps(answer)
	else:
		return'ghjk'

if __name__ == '__main__':
	data2,model2=loaddata()
	app.run(host= '59.78.8.51',port=6666,debug=True)
	#app.run(debug=True)
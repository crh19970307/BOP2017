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
		global model2,data2,vec_model
		#print(model2)
		answer={}
		answer['ans']=getanswer(dataDict['que'],data2,model2,vec_model)
		print (json.dumps(answer))
		return json.dumps(answer)
	else:
		return'ghjk'

if __name__ == '__main__':
            data2,model2=loaddata()
            vec_model = Word2Vec.load('Word60.model')
            app.run(host= '10.0.0.4',port=6666,debug=True)
	#app.run(debug=True)
from flask import Flask, jsonify,request
import numpy as np 
import re
import jieba
import jieba.analyse 
import sys
import math
import copy
from gensim.models import Word2Vec
import json
import tensorflow as tf 
from keras.models import load_model
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
    f2=open('baike.txt', 'r', encoding='utf-8')
    r2=f2.read()
    r2 = re.sub(r'|·|●|★|\*|>', '', r2)
    data2=re.sub(r'\s+', ' ', r2)



    whole_data=re.findall(r'(?<=[,^，\s！？。；;!?])(?=([^,，\s！？。；;!?]+[，,\s！？。；;!?][^,，\s！？。；;!?]*))',data2)

    model = load_model('0.608113.h5')
    global graph
    graph = tf.get_default_graph()
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


def checkQuestion(question):
    return '吗' in question or '是不是' in question or ( '么' in question and '什么' not in question)

def checkForEmission(question):
    return len(question)<=6 or '那' in  question



def getanswer(question, data, model,vec_model,oldQuestion,oldAns):
    #print(0)
    if '上海交通大学' in question:
    	question=re.sub('上海交通大学','',question)
    if '？' not in question:
    	question=question+'？'
    #print(question)
    serialTalk=checkForEmission(question)
    if serialTalk:
        q1 = jieba.lcut_for_search(oldQuestion)
        q2 = jieba.lcut_for_search(question)
        q=''
        for word2 in q2:
           if word2 not in q1 and word2 not in ['那','那么']:
               q=q+word2
        question=oldQuestion[0:-1]+q+q+'？'
        #print('this is new question',question)
    ans=getNormalAnswer(question,data,model,vec_model,serialTalk)
    #print('===+++',ans,len(ans)>1,serialTalk,ans[0]==oldAns,oldAns)
    ans=ans[1] if len(ans)>1 and serialTalk and ans[0]==oldAns  else ans[0]
    #print(2.9)
    if checkQuestion(question):
        cnt=0
        for qword in question:
            if qword in vec_model.vocab:
                vec1 = vec_model[qword]
                asentence = jieba.lcut_for_search(ans)
                for aword in asentence:
                    if aword in vec_model.vocab:
                        vec2 = vec_model[aword]
                        sim = ifSimilar(vec1, vec2)
                        if sim > 0.95:  # 0.9
                            if qword !=aword  and qword not in ['是', '的', '，', '？']:
                                cnt = cnt + sim ** 24 # 30
                                continue
            if qword in ans and qword not in ['是', '的', '，', '？']:
                cnt = cnt + 1
        #print(3.1)

        if cnt+0.1-len(ans)**2/100>2.3:
            return '是', question
        else:
            return '否',question
    else:
        #print(3)
        return ans,question



def getNormalAnswer(question, data, model,vec_model,serial):
    accurateList=['教授','电话','网址','邮箱','校长','副校长','院长','副院长','党委书记','常委','主任','系主任']
    numberofpossibleanswer = 3
    approximateSearchRate=0.9
    lengthPenaltyRate=0.25
    approximateThreshold=0.86
    seg_list = jieba.lcut_for_search(question)

    for i in seg_list:
        if i in accurateList:
            numberofpossibleanswer=1 if not serial else 2
    cntlist = []
    ############################
    ####################################
    # 第一步删选
    englishWord=re.compile('\d|[a-z]|[A-Z]|\.|（|）|-|是|的|@')

    for index, item in enumerate(data):
        cnt = 0
        for item2 in seg_list:
            if item2 in vec_model.vocab and item2 not in accurateList:
                vec1 = vec_model[item2]
                sentence = jieba.lcut_for_search(item)
                for item3 in sentence:
                    if item3 in vec_model.vocab:
                        vec2 = vec_model[item3]
                        sim = ifSimilar(vec1, vec2)
                        if sim > approximateThreshold:  # 0.9
                            if item2 != item3 and item2 not in ['是', '的', '，', '？']:
                                cnt = cnt + sim ** (30*approximateSearchRate)  # 24
                                continue
            if item2 in item and item2 not in ['是', '的', '，', '？']:
                cnt = cnt + 1
        englishWordNum=len(englishWord.findall(item))
        if len(item)-englishWordNum != 0:
            #if cnt >= 5:
            #    print('=======',cnt, (len(item)-englishWordNum) ** lengthPenaltyRate, cnt / ((len(item)-englishWordNum) ** lengthPenaltyRate),item)
            cntlist.append(cnt / ((len(item)-englishWordNum) ** lengthPenaltyRate))  # 0.25
        else:
            cntlist.append(0)
    ###############################


    cntarray = np.array(cntlist)
    indexlist = np.argsort(cntarray)
    indexlist = indexlist.tolist()
    indexlist.reverse()
    sentencelist = []
    ##########################
    # 适当切分再删选
    # print(numberofpossibleanswer)

    for i in range(numberofpossibleanswer):
        item = data[indexlist[i]]
        if len(item) >= 3:
            sentencelist.append(item)
            if (',' in item or '，' in item or ' ' in item) and re.match(r'.*“.+[,，\s].+”.*', item) is None:
                a = re.split('[,，\s]', item)
                for item2 in a:
                    if item2 not in sentencelist:
                        words1, words2 = jieba.lcut_for_search(question), jieba.lcut_for_search(item2)
                        for word1 in words1:
                            if word1 in words2 and word1 not in ['是', '的', '，', '？']:
                                sentencelist.append(item2)
    #print(2.4)
    sentencelist = list(set(sentencelist))
    answerlist=copy.deepcopy(sentencelist)
    for i,val in enumerate(sentencelist):
        sentencelist[i]=question+sentencelist[i]
    #print(2.5)
    #########################
    wordlist = spiltword(sentencelist)
    vectorlist = turnwordtovector(wordlist, vec_model)
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
    #print(2.6)
    #print(answerlist[i] for i in result[0:3])
    return [answerlist[i] for i in result[0:2]]



app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("index.html")

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
		global oldQue,ans
		ans,oldQue =getanswer(dataDict['que'], data2, model2,vec_model,oldQue,ans)
		answer['ans']=ans
		print(dataDict['que'])
		print (answer['ans'])
		return json.dumps(answer)
	else:
		return'ghjk'

if __name__ == '__main__':
			data2,model2=loaddata()
			vec_model = Word2Vec.load('Word60.model')
			oldQue=''
			ans=''
			app.run(host= '10.0.0.4',port=6666,debug=True)
	#app.run(debug=True)
import numpy as np 
import re
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
	wordlist=[]
	#print('\nSplitting word\n')
	for index,storage in enumerate(sentencelist):
		wordsplit=list(jieba.cut(storage,cut_all=False))
		wordlist.append(wordsplit)
	return wordlist 

def turnwordtovector(wordlist):
	#print('\nLoading word2vec model...\n')
	model=Word2Vec.load('Word60.model')
	#print('\nTurning word to vector\n')
	vectorlist=[]
	for index,data in enumerate(wordlist):
		tmp=[]
		for item in data:
			if item in model.vocab:
				tmp.append(model[item].tolist())
		vectorlist.append(tmp)
		#progressbar(index,len(wordlist))
	return vectorlist
def loaddata():
	f=open('baike.txt','r',encoding='utf-8')
	r=f.read()
	#print(r[0])
	data=re.sub(r'\s+', ' ', r)
	data=re.split('[！？。；;!?]',data)
	model=load_model('model/0.608113.h5')
	return data,model
	'''
	f2=open('baike-process.txt','w')
	f2.write(str(data))
	f2.close()
	'''
	#print(data[0:10])
def getanswer(question,data,model):
	seg_list = jieba.lcut_for_search(question)
	#print(seg_list)
	numberofpossibleanswer=5
	cntlist=[]
	for index,item in enumerate(data):
		cnt=0
		for item2 in seg_list:
			if item2 in item:
				cnt=cnt+1
				#print('xcvxcv')
		cntlist.append(cnt)
	cntarray=np.array(cntlist)
	#print(cntarray)
	indexlist=np.argsort(cntarray)
	indexlist=indexlist.tolist()
	#print(indexlist)
	indexlist.reverse()
	#print(indexlist)
	sentencelist=[]
	answerlist=[]
	for i in range(numberofpossibleanswer):
		#print(data[indexlist[i]])	
		tmp=re.split('[,，]',data[indexlist[i]])
		for item in tmp:
			sentencelist.append(question+item)
			answerlist.append(item)
	wordlist=spiltword(sentencelist)
	vectorlist=turnwordtovector(wordlist)
	xa=np.zeros((len(vectorlist),30,60),dtype='float64')
	for index1,items in enumerate(vectorlist):
		for index2,item2 in enumerate(items):
			if index2==30:
				break
			xa[index1][index2]=item2
	result=model.predict(xa)
	#print(result)
	tmp=[]
	for i in range(len(vectorlist)):
		tmp.append(result[i][0])
	result=np.array(tmp)
	result=np.argsort(result)
	result=result.tolist()
	result.reverse()
	#print(data[indexlist[result[0]]])
	return sentencelist[result[0]]+'\n'+sentencelist[result[1]]
	'''
	for i in range(numberofpossibleanswer):
		print('\n')
		print(tmp[result[i]])
		print(data[indexlist[result[i]]])
		'''
def main():
	data,model=loaddata()
	questionlist=['请问贵校的校长是谁？','请问贵校建校于哪一年？','贵校有多少本科专业？','你们的校歌是什么？','你们学校坐落于哪座城市/省份？','你们学校有多少工程院院士？','贵校有多少学院？','请问贵校的校长是谁？']
	for question in questionlist:
		print('\n')
		print(question)
		print(getanswer(question,data,model))
	#wordsplit=list(jieba.cut(question,cut_all=False))
	
if __name__=='__main__':
	main()

	
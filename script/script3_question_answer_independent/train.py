import numpy as np 
import re
import jieba
import jieba.analyse 
import sys
import math
from gensim.models import Word2Vec
import json
from keras.preprocessing import sequence  
from keras.optimizers import SGD, RMSprop, Adagrad  
from keras.utils import np_utils  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.layers.embeddings import Embedding  
from keras.layers.recurrent import LSTM, GRU  
from keras.models import load_model

def getdata():
	f=open('train.txt','r',encoding='utf-8')
	#f2=open('score.txt','w')
	r=f.readlines()
	tmpquestion=" "
	quecnt=-1
	quelist=[] #question as index, point to the number whose answer is related to this question
	answerlist=[] #question as index, point to the correct answer's number
	datalist=[] #datalist is the list of origin data, datalist[i][0] is the correctness of question-answer pair i. datalist[i][1]is the question ,data;ist[i][2] is the answer
	print('Loading data\n')
	for cnt,lines in enumerate(r):
		l=re.sub(r'\s+', ' ', lines) #Replace the multy space with one space
		l=l.split(' ',2) #l contains 3 parts after this operation
		if l[0]!='0' and l[0]!='1':
			l[0]='0'
		datalist.append(l)
		#f2.write(l[0])
		#f2.write('\n')
		if l[1] !=tmpquestion:
			quecnt=quecnt+1
			tmpquestion=l[1]
			quelist.append([])
			answerlist.append([])
		if l[0]=='1':
			answerlist[quecnt]=cnt
		quelist[quecnt].append(cnt)
		progressbar(cnt,len(r))
	#f3.write(str(answerlist))
	#print(datalist)
	f.close()
	#f2.close()
	return quelist,answerlist,datalist
	#f2.close()
	#f3.close()

def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %s" % (
                            '=' * int(math.floor(cur * 50 / total)),
                            percent))
    sys.stdout.flush()

def spiltword(datalist):
	#f=open('wordlist.txt','w',encoding='utf-8')
	wordlist1=[]
	wordlist2=[]
	print('\nSplitting word\n')
	for index,storage in enumerate(datalist):
		wordsplit1=list(jieba.cut(storage[1],cut_all=False))
		wordsplit2=list(jieba.cut(storage[2],cut_all=False))
		#wordsplit=jieba.analyse.extract_tags(storage[1],8)+jieba.analyse.extract_tags(storage[2],8)
		wordlist1.append(wordsplit1)
		wordlist2.append(wordsplit2)
		#f.write(str(wordsplit))
		#f.write('\n')
		progressbar(index,len(datalist))
		#print( storage[1]+' '+storage[2])
		#print(wordsplit)
		#print('\n')
		#if index==2:
		#	break
	#print ((wordlist))
	#f.close()
	return wordlist1,wordlist2 

def turnwordtovector(wordlist1,wordlist2):
	print('\nLoading word2vec model...\n')
	model=Word2Vec.load('Word60.model')
	print('\nTurning word to vector\n')
	vectorlist1=[]
	vectorlist2=[]
	for index,data in enumerate(wordlist1):
		tmp=[]
		for item in data:
			if item in model.vocab:
				tmp.append(model[item].tolist())
				#print(list(model[item]))
				#print('\n')
		vectorlist1.append(tmp)
		progressbar(index,len(wordlist1))

	for index,data in enumerate(wordlist2):
		tmp=[]
		for item in data:
			if item in model.vocab:
				tmp.append(model[item].tolist())
				#print(list(model[item]))
				#print('\n')
		vectorlist2.append(tmp)
		progressbar(index,len(wordlist2))
	#f=open("wordlist3.txt",'w',encoding='utf-8')
	#f.write(str(vectorlist))
	#f.close()
	return vectorlist1,vectorlist2

def savetojson():
	f=open("data_v2.json","w",encoding='utf-8')
	quelist,answerlist,datalist=getdata()
	wordlist1,wordlist2=spiltword(datalist)
	vectorlist1,vectorlist2=turnwordtovector(wordlist1,wordlist2)
	python2json={}
	python2json["quelist"]=quelist
	python2json["answerlist"]=answerlist
	python2json["datalist"]=datalist
	python2json["wordlist1"]=wordlist1
	python2json["wordlist2"]=wordlist2
	python2json["vectorlist1"]=vectorlist1
	python2json["vectorlist2"]=vectorlist2
	jsonstr=json.dumps(python2json)
	#f=open("30000data.json","w",encoding='utf-8')
	f.write(jsonstr)
	f.close()

def loadfromjson():
	f=open('data_v2.json','r',encoding='utf-8')
	data=json.load(f)
	return data

def train():
	import os
	gpu_id = '0'
	os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
	os.system('echo $CUDA_VISIBLE_DEVICES')
	data=loadfromjson()
	taglist=[]
	for index,item in enumerate(data['datalist']):
		if item[0]=='0':
			taglist.append(0)
		else:
			if item[0]=='1':
				taglist.append(1)
			else:
				print('EiRROR\n')
				print(index)
				taglist.append(0)
		#print(len(data['vectorlist'][index]))
	#xa=np.array(data['vectorlist'])
	xa=np.zeros((len(data['vectorlist1']),45,60),dtype='float64')
	for index1,items in enumerate(data['vectorlist1']):
		for index2,item2 in enumerate(items):
			if index2==15:
				break
			xa[index1][index2]=item2
	for index1,items in enumerate(data['vectorlist2']):
		for index2,item2 in enumerate(items):
			if index2==30:
				break
			xa[index1][index2+15]=item2		

	#xa=np.random.rand(len(data['vectorlist']),50,60)
	ya=np.array(taglist)
	#print(np.size(xa))
	#print(np.size(ya))
	print('Build model...')  
	model = Sequential()  
	#model.add(Embedding(60,32))  
	#model.add(LSTM(128)) # try using a GRU instead, for fun  
	#model.add(LSTM(32,input_shape=(10,60)))
	#model.add(LSTM(32,input_length=50,input_dim=60))
	model.add(LSTM(128,input_length=45,input_dim=60))
	print('LSTM added')
	model.add(Dropout(0.5))  	
	model.add(Dense( 1))  
	model.add(Activation('sigmoid'))  
	model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")  
	model.fit(xa, ya, batch_size=128, nb_epoch=100) #训练时间为若干个小时  
	model.save('sentencelen15+30_230000_lstm128_epoch100_model.h5')

def continue_train():
	model=load_model('my_model.h5')
	data=loadfromjson()
	taglist=[]
	for index,item in enumerate(data['datalist']):
		if item[0]=='0':
			taglist.append(0)
		else:
			if item[0]=='1':
				taglist.append(1)
			else:
				print('ERROR')
				print(index)
				taglist.append(0)
		#print(len(data['vectorlist'][index]))
	#xa=np.array(data['vectorlist'])
	xa=np.zeros((len(data['vectorlist']),45,60),dtype='float64')
	for index1,items in enumerate(data['vectorlist']):
		for index2,item2 in enumerate(items):
			if index2==20:
				break
			xa[index1][index2]=item2

	#xa=np.random.rand(len(data['vectorlist']),50,60)
	ya=np.array(taglist)
	model.fit(xa, ya, batch_size=1024, nb_epoch=50) #训练时间为若干个小时  
	model.save('my_model2.h5')

if __name__=='__main__':
	#getdata()
	#savetojson()
	#continue_train()
	train()

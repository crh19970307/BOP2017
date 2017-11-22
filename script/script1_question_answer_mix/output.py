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
	f=open('dev.txt','r',encoding='utf-8')
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
	#	if cnt==30000:
	#		break
		progressbar(cnt,len(r))
	#f3.write(str(answerlist))
	#print(datalist)
	f.close()
	#f2.close()
	return quelist,answerlist,datalist
	#f2.close()
	#f3.close()
def calculateMRR(quelist,answerlist,datalist):
	f=open('score.txt','r',encoding='utf-8')
	r=f.readlines()
	score=[]
	for i in r:
		score.append(float(i))
	sum=0
	for cnt,question in enumerate(quelist):
		tmp=[]
		for i in question:
			tmp.append(score[i])
		tmp.sort()
		#print(cnt)
		if answerlist[cnt]==[]:
			#This case exists no true answer, the score is 0
			pass
		else:
			tmp.reverse()
			#if tmp.index(score[answerlist[cnt]])!=0:
			#	print(tmp)
			sum=sum+1/(1+tmp.index(score[answerlist[cnt]]));
	print('\n')	
	print(sum/len(quelist))

def randomscore():
	#generate a random score, the MRR of it is approximately 0.26
	f=open('test.txt','w',encoding='utf-8')
	for i in range(99110):
		f.write(str(np.random.rand()))
		f.write('\n')
	f.close()

def progressbar(cur, total):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %s" % (
                            '=' * int(math.floor(cur * 50 / total)),
                            percent))
    sys.stdout.flush()

def spiltword(datalist):
	#f=open('wordlist.txt','w',encoding='utf-8')
	wordlist=[]
	print('\nSplitting word\n')
	for index,storage in enumerate(datalist):
		wordsplit=list(jieba.cut(storage[1]+' '+storage[2],cut_all=False))
		#wordsplit=jieba.analyse.extract_tags(storage[1],8)+jieba.analyse.extract_tags(storage[2],8)
		wordlist.append(wordsplit)
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
	return wordlist 

def loadsplitword():
	f=open('wordlist.txt','r',encoding='utf-8')
	wordlist=[]
	r=f.readlines()
	for index,data in enumerate(r):
		#print(i)
		tmp=eval(str(data))
		#print(tmp)
		wordlist.append(tmp)
		progressbar(index,len(r))
	return wordlist

def turnwordtovector(wordlist):
	print('\nLoading word2vec model...\n')
	model=Word2Vec.load('Word60.model')
	print('\nTurning word to vector\n')
	vectorlist=[]
	for index,data in enumerate(wordlist):
		tmp=[]
		for item in data:
			if item in model.vocab:
				tmp.append(model[item].tolist())
				#print(list(model[item]))
				#print('\n')
		vectorlist.append(tmp)
		progressbar(index,len(wordlist))
	#f=open("wordlist3.txt",'w',encoding='utf-8')
	#f.write(str(vectorlist))
	#f.close()
	return vectorlist

def savetojson():
	quelist,answerlist,datalist=getdata()
	wordlist=spiltword(datalist)
	vectorlist=turnwordtovector(wordlist)
	python2json={}
	python2json["quelist"]=quelist
	python2json["answerlist"]=answerlist
	python2json["datalist"]=datalist
	python2json["wordlist"]=wordlist
	python2json["vectorlist"]=vectorlist
	jsonstr=json.dumps(python2json)
	f=open("3000data.json","w",encoding='utf-8')
	f.write(jsonstr)
	f.close()

def loadfromjson():
	f=open('data.json','r',encoding='utf-8')
	data=json.load(f)
	return data

def train():
	data=loadfromjson()
	taglist=[]
	for index,item in enumerate(data['datalist']):
		if item[0]=='0':
			taglist.append(0)
		else:
			if item[0]=='1':
				taglist.append(1)
			else:
				print('ERROR\n')
				print(index)
		#print(len(data['vectorlist'][index]))
	#xa=np.array(data['vectorlist'])
	xa=np.zeros((len(data['vectorlist']),50,60),dtype='float64')
	for index1,items in enumerate(data['vectorlist']):
		for index2,item2 in enumerate(items):
			if index2==50:
				break
			xa[index1][index2]=item2

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
	model.add(LSTM(32,input_dim=60))
	print('LSTM added')
	model.add(Dropout(0.5))  	
	model.add(Dense( 1))  
	model.add(Activation('sigmoid'))  
	model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")  
	model.fit(xa, ya, batch_size=16, nb_epoch=100) #训练时间为若干个小时  
	model.save('my_model.h5')

def output():
	model=load_model('sentencelen30_230000_lstm128_epoch200_model.h5')
	quelist,answerlist,datalist=getdata()
	wordlist=spiltword(datalist)
	vectorlist=turnwordtovector(wordlist)
	f=open('score.txt','w',encoding='utf-8')
	xa=np.zeros((len(vectorlist),30,60),dtype='float64')
	for index1,items in enumerate(vectorlist):
		for index2,item2 in enumerate(items):
			if index2==30:
				break
			xa[index1][index2]=item2
	result=model.predict(xa)
	print(result)
	print(type(result))
	for i in result:
		f.write(str((i[0])))
		f.write('\n')
	f.close()
if __name__=='__main__':
	#getdata()
	#savetojson()
	output()
	quelist,answerlist,datalist=getdata()
	calculateMRR(quelist,answerlist,datalist)

import numpy as np 
import re

def getdata():
	f=open('dev.txt','r',encoding='utf-8')
	#f2=open('score.txt','w')
	r=f.readlines()
	tmpquestion=" "
	quecnt=-1
	quelist=[] #question as index, point to the number whose answer is related to this question
	answerlist=[] #question as index, point to the correct answer's number
	datalist=[] #datalist is the list of origin data, datalist[i][0] is the correctness of question-answer pair i. datalist[i][1]is the question ,data;ist[i][2] is the answer
	for cnt,lines in enumerate(r):
		l=re.sub(r'\s+', ' ', lines) #Replace the multy space with one space
		l=l.split(' ',2) #l contains 3 parts after this operation
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
	#f3.write(str(answerlist))
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
	print(sum/len(quelist))

def randomscore():
	#generate a random score, the MRR of it is approximately 0.26
	f=open('test.txt','w',encoding='utf-8')
	for i in range(99110):
		f.write(str(np.random.rand()))
		f.write('\n')
	f.close()

if __name__=='__main__':
	quelist,answerlist,datalist=getdata()
	calculateMRR(quelist,answerlist,datalist)
	
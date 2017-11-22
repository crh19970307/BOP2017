from flask import Flask, jsonify,request
import numpy as np 
import re
import jieba
import jieba.analyse 
import sys
import math
import copy
import json



def spiltword(sentencelist):
    wordlist = []
    for index, storage in enumerate(sentencelist):
        wordsplit = list(jieba.cut(storage, cut_all=False))
        wordlist.append(wordsplit)
    return wordlist


def loaddata():
    f2=open('baike.txt', 'r', encoding='utf-8')
    r2=f2.read()
    r2 = re.sub(r'|·|●|★|\*|>', '', r2)
    data2=re.sub(r'\s+', ' ', r2)
    whole_data=re.findall(r'(?<=[,^，\s！？。；;!?])(?=([^,，\s！？。；;!?]+[，,\s！？。；;!?][^,，\s！？。；;!?]*))',data2)
    tmp=[]
    for i,item in enumerate(whole_data):
        if len(re.split('[！？。；;!?]', item)) > 1:
            t=re.split('[！？。；;!?]', item)
            for i in t:
                tmp.append(i)
        else:
            tmp.append(item)
    whole_data=list(set(tmp))
    return whole_data

def getanswer(question,data):
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
    
    return answerlist[0]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'exciting!!!'

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
        print('\nrequest.form is :\n')
        print(request.form)
        print('\nrequest.args is:\n')
        print(request.args)
        print('\nrequest.value is:\n')
        print(request.values)
        print('\nrequest.cookies is\n')
        print(request.cookies)
        # #dataDict = request.data 
        print('\nrequest.data is :\n')
        print(request.data)
        #dataDict=json.loads(request.data.decode())
        dataDict=request.form
        #print(dataDict)
        #print(dataDict)
        #return str(dataDict)
        #return 'hdlkgjahbadks'
        global data2
        #print(model2)
        answer={}
        
        ans =getanswer(dataDict['que'], data2)
        answer['ans']=ans
        print(dataDict['que'])
        print (answer['ans'])
        return json.dumps(answer)
    else:
        return'ghjk'

if __name__ == '__main__':
            data2=loaddata()
          
            app.run(host= '10.0.0.4',port=8000,debug=True)
            #app.run(debug=True)
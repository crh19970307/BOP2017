from flask import Flask, jsonify,request
import numpy as np 
import re
import jieba
import jieba.analyse 
import sys
import math
from gensim.models import Word2Vec
import json
from keras.models import load_model


tasks = [
	{
		'id': 1,
		'title': u'OSPA',
		'description': u'This is ospaf-api test', 
		'done': False
	},
	{
		'id': 2,
		'title': u'Garvin',
		'description': u'I am garvin', 
		'done': False
	}
]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return jsonify({'tasks': tasks})


@app.route('/post/', methods=['GET', 'POST'])
def real_time_api():
	if request.method == 'POST':
		#question=request.data
		#question=request.data
		data2,model2=loaddata()
		datadict=request.form
		print(datadict['que'])
		#print(data2)
		#return datadict['que']
		
		print(request.form)
		print(request.args)
		print(request.values)
		print(request.cookies)
		#dataDict = request.data 
		print(request.data)
		
		#print(dataDict)
		#print(dataDict)
		#return str(dataDict)
		#return 'hdlkgjahbadks'
		return datadict['que']
	else:
		return'ghjk'

if __name__ == '__main__':
	#app.run(host= '59.78.8.51',port=6666,debug=True)
	
	app.run(debug=True)
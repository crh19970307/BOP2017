'''
import urllib
from urllib import request,parse 
#定义一个要提交的数据数组(字典)
data = {}
data['username'] = 'zgx030030'
data['password'] = '123456'
 
#定义post的地址
url =  'http://127.0.0.1:5000/post/'
post_data = parse.urlencode(data).encode('utf-8')
print(post_data)
print('\n')
#提交，发送数据
req = request.urlopen(url, post_data)
 
#获取提交后返回的信息
content = req.read()
print(content)
'''
import requests
import json 
payload = {'que': '贵校有多少本科专业？'}
r = requests.post("http://42.159.206.59:8000/post/", data=payload)
#r=requests.post('http://59.78.8.51:5000/post/', data={'number': 12524, 'type': 'issue', 'action': 'show'})
print(json.loads(r.text)['ans'])
'''
payload = {'que': '你们学校有多少工程院院士？'}
r = requests.post("http://59.78.8.51:6666/post/", data=payload)
#r=requests.post('http://59.78.8.51:5000/post/', data={'number': 12524, 'type': 'issue', 'action': 'show'})
print(r.text)
'''
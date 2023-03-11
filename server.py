# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template, request
import json
import AI_methods_20221020 as ai
import test6 
app = Flask(__name__)  # 实例化app对象

testInfo = {}


@app.route('/test_post/nn', methods=['GET', 'POST'])  # 路由
def test_post():
    # url = request.stream.read()

    # data = request.get_data()
    # data = json.loads(data)
    #
    # srxApi = data['srxApi']
    # operate = data['operate']
    # print(srxApi+"+"+operate)



    # get data from ajax in .js
    url = request.form['url']
    print("url: "+url)

    tmp = test6.qw(url)
    # ai.test_ai(url)

    # ai.test_print(url)

    testInfo['name'] = 'xiaoming'
    testInfo['age'] = '18' 
    
    return json.dumps(tmp)

@app.route('/testpost/nn', methods=['GET', 'POST'])  # 路由
def testpost():
    url = request.form['url']
    # print("url: "+url)
    # ai.test_ai(url)

    tmp=ai.acc(url) 
    return json.dumps(tmp)


@app.route('/testpost2/nn', methods=['GET', 'POST'])  # 路由
def testpost2():
    url = request.form['url']
    # print("url: "+url)
    # ai.test_ai(url)

    tmp=ai.acc2(url) 
    return json.dumps(tmp)

@app.route('/testpost3/nn', methods=['GET', 'POST'])  # 路由
def testpost3():
    url = request.form['url']
    # print("url: "+url)
    # ai.test_ai(url)

    tmp=ai.acc3(url) 
    return json.dumps(tmp)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/index2')
def index2():
    return render_template('pnetedit.html')


@app.route('/index3')
def index3():
    return render_template('pnetedit2.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',  # 任何ip都可以访问
            port=7777,  # 端口
            debug=True
            )

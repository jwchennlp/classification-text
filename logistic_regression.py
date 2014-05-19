#!/usr/bin/env python
#encoding:utf-8

'''
function:实现逻辑回归
author:jwchen
date:2014-05-18
'''
import pickle
import numpy as np
import naive_bayes as nb
import text_vector as vec
from sklearn.linear_model import LogisticRegression
import random

def preprocess():
    traindir = './data/training'
    testdir = './data/test'
    
    tokens_all_x = nb.read('tokens_all_x')
    train_x,train_y,category = nb.func2(traindir)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    result,test_x,test_file = nb.func3(testdir,category)
    test_x = np.array(test_x)
    test_file = np.array(test_file)
    
    nb.write(train_x,'train_x')
    nb.write(train_y,'train_y')
    nb.write(category,'category')
    nb.write(result,'result')
    nb.write(test_x,'test_x')
    nb.write(test_file,'test_file')

#用sklearn工具包实现逻辑回归
def logistic_l1():
    traindir ='./data/training'
    testdir = './data/test'
    
    tokens = list(nb.read('tokens'))
    train_x,train_y,category = vec.func2(traindir,tokens)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print train_x.shape
    clf = LogisticRegression(penalty='l2')
    clf.fit(train_x,train_y)

    category = nb.read('category')
    result,test_x,test_file= vec.func3(testdir,tokens,category)
    test_x = np.array(test_x)
    print test_x.shape
    
    predict = np.array(clf.predict(test_x))
    
    test_file = np.array(test_file)
    predict = np.column_stack((test_file,predict))
    
    category = nb.read('category_nb_eventmodel')
    category_convert = nb.convert(category)
    result = nb.read('result')
    path = './data/logistic_l1.csv'
    evaluate = nb.sta_result(predict,category_convert,result,path)
#vector space model and select feature through x^2
def logistic_x():
    train_x,train_y,category,result,test_x,test_file = preprocess()
    clf = LogisticRegression(penalty='l1')
    clf.fit(train_x,train_y)
    
    predict = clf.predict(test_x)
    
    predict = np.array(predict)
    predict = np.column_stack((test_file,predict))
    
    category = nb.read('category_nb_eventmodel')
    category_convert = nb.convert(category)
    result = nb.read('result')
    path = './data/logistic_l1.csv'
    evaluate = nb.sta_result(predict,category_convert,result,path)
    
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))
def calj(binary_y,h,m):
    j = 0
    for index in range(len(binary_y)):
        if binary_y[index] == 1:
            if h[index] == 0:
                j += 50
            else:
                j += -np.log2(h[index])
        else:
            if h[index] == 1:
                j += 50
            else:
                j += -np.log2(1-h[index])
    j = j/float(m)
    return j
def sto_logistic():
    train_x = nb.read('train_x')
    train_y = nb.read('train_y')
    category = nb.read('category')
    result =nb.read('result')
    test_x = nb.read('test_x')
    test_file = nb.read('test_file')
    m,n=train_x.shape
    temp = np.ones((m,1))
    train_x = np.column_stack((temp,train_x))
    
    temp = np.ones((len(test_x),1))
    test_x = np.column_stack((temp,test_x))
    
    predict = np.zeros((len(test_x),1))
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    test_x = np.mat(test_x)
    #由于要实现多分类，我们可以通过多个二分类来实现预测
    for i in range(10):
        binary_y = np.mat(np.zeros((m,1)).astype(int))
        for index in range(len(train_y)):
            if train_y[index]==i:
                binary_y[index]=1
            else:
                binary_y[index]=0
        weight = np.mat(np.ones((n+1,1)))
        alpha = 0.001
        maxitem =5000
        for k in range(maxitem):
            index = random.randrange(m)
            h = sigmoid(train_x[index]*weight)
            error = h - binary_y[index]
            weight -= alpha*(train_x[index].transpose()*error)
        binary_predict = test_x*weight
        for index in range(len(binary_predict)):
            if binary_predict[index]>0:
                predict[index]=i

    predict = np.array(predict).astype(int)
    test_file = np.array(test_file)
    predict = np.column_stack((test_file,predict))
    
    category = nb.read('category_nb_eventmodel')
    category_convert = nb.convert(category)
    result = nb.read('result')
    path = './data/logistic_l1.csv'
    evaluate = nb.sta_result(predict,category_convert,result,path)
            
def logistic_own():
    train_x = nb.read('train_x')
    train_y = nb.read('train_y')
    category = nb.read('category')
    result =nb.read('result')
    test_x = nb.read('test_x')
    test_file = nb.read('test_file')
    m,n=train_x.shape
    temp = np.ones((m,1))
    train_x = np.column_stack((temp,train_x))
    
    temp = np.ones((len(test_x),1))
    test_x = np.column_stack((temp,test_x))
    
    predict = np.zeros((len(test_x),1))
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    test_x = np.mat(test_x)
    #由于要实现多分类，我们可以通过多个二分类来实现预测
    for i in range(10):
        binary_y = np.mat(np.zeros((m,1)).astype(int))
        for index in range(len(train_y)):
            if train_y[index]==i:
                binary_y[index]=1
            else:
                binary_y[index]=0
        weight = np.mat(np.ones((n+1,1)))
        alpha = 0.0001
        maxitem = 100
        for k in range(maxitem):
            h = sigmoid(train_x*weight)
            #我们在计算代价函数的时候,不能简单的用公式实现,应当进行判断
            J = calj(binary_y,h,m)
            #J = 1.0/m*(-binary_y.transpose()*np.log2(h)-(1-binary_y.transpose())*np.log2(1-h))
            error = h-binary_y
            weight -= alpha*(train_x.transpose()*error)
        binary_predict = test_x*weight
        for index in range(len(binary_predict)):
            if binary_predict[index]>0:
                predict[index]=i

    predict = np.array(predict).astype(int)
    test_file = np.array(test_file)
    predict = np.column_stack((test_file,predict))
    
    category = nb.read('category_nb_eventmodel')
    category_convert = nb.convert(category)
    result = nb.read('result')
    path = './data/logistic_l1.csv'
    evaluate = nb.sta_result(predict,category_convert,result,path)

if __name__=="__main__":
    choice = raw_input('1.logistic regression in sklearn with l1\n2. logistic regression in vector space model with x^2\n3.logistic regression on my own way\n4.preprocess\n5.stochastic gradient descent\n')
    if choice == str(1):
        logistic_l1()
    elif choice == str(2):
        logistic_x()
    elif choice == str(3):
        logistic_own()
    elif choice == str(4):
        preprocess()
    elif choice == str(5):
        sto_logistic()

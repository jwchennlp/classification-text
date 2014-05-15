#!/usr/bin/env python
#coding:utf-8

'''
function:特征选择，互信息，x^2统计量，词频统计
author:jwchen
date:2014-05-03
'''

import os
import nltk
import pickle
import naive_bayes as nb 
import numpy as np
import math,csv




#获取每一个类别的词数
def get_category_tokens():
    train = nb.read('train_nb')
    train = np.array(train)
    category_tokens = {}
    for i in range(10):
        train_category = train[train[0::,2]==i,0]
        tokens = []
        for token in train_category:
            tokens += token
        tokens = set(tokens)
        category_tokens[i] = tokens
    return category_tokens
#互信息选取特征
def mi(train,sta,category_tokens):
    train = np.array(train)
    token_mi = {}
    for i in range(10):
        token_mi[i] = {}
        #计算每个类别下词的互信息
        for token in category_tokens[i]:
            #统计词和类别之间的关系
            N = sta_token_category(token,train,i)
            a = sum(sum(N))
            ct = N[1][0]+N[1][1]
            cf = N[0][1]+N[0][0]
            tt = N[0][1]+N[1][1]
            tf = N[0][0]+N[1][0]
            token_mi[i][token] = \
N[1][1]/float(a)*math.log((a*N[1,1]+1)/float(ct*tt),2)+ \
N[0,0]/float(a)*math.log((a*N[0,0]+1)/float(cf*tf),2)+ \
N[1,0]/float(a)*math.log((a*N[1,0]+1)/float(ct*tf),2)+ \
N[0,1]/float(a)*math.log((N[0,1]*a+1)/float(cf*tt),2)
    return token_mi

#统计词和类别之间的关系
def sta_token_category(token,train,i):
    N = np.zeros((2,2))
    t_category = train[train[0::,2]==i,0]
    f_category = train[train[0::,2]!=i,0]
    t_in_count = cal_count(token,t_category)
    t_notin_count = len(t_category) - t_in_count
    f_in_count = cal_count(token,f_category)
    f_notin_count = len(f_category) -f_in_count

    N[1][1] = t_in_count
    N[1][0] = t_notin_count
    N[0][0] = f_notin_count
    N[0][1] = f_in_count
    return N

#卡方统计
def x_2()
def cal_count(token,category):
    count = 0
    for tokens in category:
        if token in tokens:
            count +=1
    return count

def mi_func():
    traindir = './data/trainint'
    testdir = './data/test'
    
    train = nb.read('train_nb_eventmodel') 
    sta = nb.sta_count(train)
    category_tokens = get_category_tokens()
    token_mi = mi(train,sta,category_tokens)
    nb.write(token_mi,'token_mi')
    
    category = nb.read('category_nb_eventmodel')
    token_mi = nb.read('token_mi')
    category_convert = nb.convert(category)
    
    mi_category = {}
    for i in range(10):
        tokens = sorted(token_mi[i],key = token_mi[i].get,reverse = True)
        mi_category[i] = tokens[:100]
    nb.write(mi_category,'mi_category')
    
    '''
    csvfile = file('./data/mi.csv','wb')
    writer = csv.writer(csvfile)
    for i in range(10):
         tokens =  sorted(token_mi[i],key= token_mi[i].get,reverse=True)[:20]
         writer.writerow([category_convert[i]])
         print category_convert[i]
         for token in tokens:
             writer.writerow([token,'%.4f'%token_mi[i][token]])
             print token,token_mi[i][token]
     '''

def x_2func():
    
if __name__=="__main__":
    traindir = 
    

    
    

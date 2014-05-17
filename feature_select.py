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
#在频度统计时，要获取的是整个文档集的词典
def get_all_tokens():
    train = nb.read('train_nb')
    train = np.array(train)
    tokens = []
    for i in range(10):
        train_category = train[train[0::,2]==i,0]
        for token in train_category:
            tokens = set(token)|set(tokens)
    tokens = set(tokens)
    print len(tokens)
    return tokens
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

#统计某一个词在类别i下的多少文档中出现
def cal_df_count(token,train):
    count = 0
    for doc in train:
        if token in doc[0].keys():
            count += 1
    return count
#频率统计
def df(train,sta,all_tokens):
    train = np.array(train)
    tokens_df = {}
    i = 1
    for token in all_tokens:
        print('遍历第%d个词'%(i))
        count = cal_df_count(token,train)
        tokens_df[token] = count
        i += 1
    return tokens_df
def cal_entropy(train):
    category = []
    for i in range(10):
        a = train[train[0::,2]==i,0]
        category.append(len(a))
    count = sum(category)
    entropy = 0
    for i in category:
        if i !=0:
            entropy += -i/float(count)*math.log(i/float(count),2)
    return entropy
#文档集关于词项t的条件熵
def cal_gaininfo(token,train):
    t_category,nott_category = [],[]
    for doc in train:
        if token in doc[0].keys():
            t_category.append(doc)
        else:
            nott_category.append(doc)
    t_category = np.array(t_category)
    nott_category = np.array(nott_category)
    entropy1 = cal_entropy(t_category)
    entropy2 = cal_entropy(nott_category)
    entropy = len(t_category)/float(len(train))*entropy1+len(nott_category)/float(len(train))*entropy2
    return entropy
#信息增益
def gi(train,sta,all_tokens):
    train = np.array(train)
    #文档集下类别的熵
    
    entropy = cal_entropy(train)
    print entropy

    tokens_gi = {}
    i = 1
    for token in all_tokens:
        print ('遍历第%d个词'%(i))
        condition_entropy = cal_gaininfo(token,train)
        tokens_gi[token] = entropy - condition_entropy
        i += 1
    return tokens_gi

    
#卡方统计
def x_2(train,sta,category_tokens):
    train = np.array(train)
    token_x = {}
    for i in range(10):
        token_x[i] = {}
        #计算每个类别下每个词的x^2检验量
        for token in category_tokens[i]:
            N = sta_token_category(token,train,i)
            a = sum(sum(N))
            ct = N[1][0]+N[1][1]
            cf = N[0][1]+N[0][0]
            tt = N[0][1]+N[1][1]
            tf = N[0][0]+N[1][0]
            #要计算词和类别相互独立的前提下出现的期望频率
            E = cal_e(a,ct,cf,tt,tf)
            token_x[i][token] = a*pow((N[1,1]*N[0,0]-N[1,0]*N[0,1]),2)/float(ct*cf*tt*tf)
            '''
            token_x[i][token] = pow(N[1,1]-E[1,1],2)/float(E[1,1])+\
                pow(N[0,0],2)/float(E[0,0])+\
                pow(N[0,1],2)/float(E[0,1])+\
                pow(N[1,0],2)/float(E[1,0])
            '''
    return token_x

def cal_e(a,ct,cf,tt,tf):
    E = np.zeros((2,2))
    E[1][1] = ct*tt/float(a)
    E[0,0] = cf*tf/float(a)
    E[1,0] = ct*tf/float(a)
    E[0,1] = cf*tt/float(a)
    return E

def cal_count(token,category):
    count = 0
    for tokens in category:
        if token in tokens:
            count +=1
    return count

def mi_func():
    '''
    train = nb.read('train_nb_eventmodel') 
    sta = nb.sta_count(train)
    category_tokens = get_category_tokens()
    token_mi = mi(train,sta,category_tokens)
    nb.write(token_mi,'token_mi')
    '''
    category = nb.read('category_nb_eventmodel')
    token_mi = nb.read('token_mi')
    category_convert = nb.convert(category)
    tokens_all_mi = []
    mi_category = {}
    for i in range(10):
        tokens = sorted(token_mi[i],key = token_mi[i].get,reverse = True)
        mi_category[i] = tokens[:500]
        tokens_all_mi = set(tokens_all_mi)|set(mi_category[i])
    print len(tokens_all_mi)
    nb.write(mi_category,'mi_category')
    nb.write(tokens_all_mi,'tokens_all_mi')
    
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

def x_2func(traindir,testdir):
    '''
    train = nb.read('train_nb_eventmodel')
    sta = nb.sta_count(train)
    category_tokens = get_category_tokens()
    token_x = x_2(train,sta,category_tokens)
    nb.write(token_x,'token_x')
    
    '''
    category = nb.read('category_nb_eventmodel')
    tokens_x = nb.read('token_x')
    category_convert = nb.convert(category)
    tokens_all_x = []
    x_category = {}
    for i  in range(10):
        tokens = sorted(tokens_x[i],key=tokens_x[i].get,reverse = True)
        x_category[i] = tokens[:100]
        '''
        x_category[i] = []
        for word in tokens:
            if tokens_x[i][word] >10.83:
                x_category[i].append(word)
            else:
                break
        '''
        print len(x_category[i])
        tokens_all_x = set(tokens_all_x)|set(x_category[i])
    print len(tokens_all_x)
    nb.write(x_category,'x_category')
    nb.write(tokens_all_x,'tokens_all_x')
    
def dffunc():
    #这里先使用文档频率，需要统计在某一类别下词t在多少个文档中出现
    '''
    train = nb.read('train_nb_eventmodel')
    sta = nb.sta_count(train)
    all_tokens = get_all_tokens()
    tokens_df = df(train,sta,all_tokens)
    nb.write(tokens_df,'tokens_df')
    '''
    category = nb.read('category_nb_eventmodel')
    tokens_df = nb.read('tokens_df')
    category_convert = nb.convert(category)
    tokens_all_df = []
    df_category = {}
    for i in range(10):
        tokens = sorted(tokens_df,key=tokens_df.get,reverse = True)
        df_category[i] = tokens[:200]
        tokens_all_df = set(tokens_all_df)|set(df_category[i])
    print tokens_all_df

    nb.write(df_category,'df_category')
    nb.write(tokens_all_df,'tokens_all_df')

#用信息增益实现特征抽取
def gifunc():
    train = nb.read('train_nb_eventmodel')
    sta = nb.sta_count(train)
    all_tokens = get_all_tokens()
    tokens_gi = gi(train,sta,all_tokens)
    nb.write(tokens_gi,'tokens_gi')

    category = nb.read('category_nb_eventmodel')
    tokens_gi = nb.read('tokens_gi')
    category_convert = nb.convert(category)
    tokens_all_gi = []
    gi_category = {}
    for i in range(10):
        tokens = sorted(tokens_gi,key=tokens_gi.get,reverse=True)
        gi_category[i] = tokens[:100]
        tokens_all_gi = set(tokens_all_gi)|set(gi_category[i])
    print tokens_all_gi
    nb.write(gi_category,'gi_category')
    nb.write(tokens_all_gi,'tokens_all_gi')
if __name__=="__main__":
    traindir = './data/training'
    testdir = './data/test'
    
    gifunc()
    

    
    

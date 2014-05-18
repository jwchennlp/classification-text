#!/usr/bin/env python
#coding:utf-8

'''
function:构建训练集和测试集,将文档按向量的方式表述
author:jwchen
date:2014-05-13
'''

import os
import nltk
import pickle
import numpy as np

def write(dic,file_name):
    pickle.dump(dic,open('./data/'+file_name,'wb'))
def read(file_name):
    dic = pickle.load(open('./data/'+file_name,'rb'))
    return dic
def get_folder_list(path):
    folderlist = os.listdir(path)
    return folderlist

def get_file_list(path,folder):
    file_path = path+'//'+folder
    files_list = os.listdir(file_path)
    return files_list
#统计训练集中的词表
def get_tokens(dir,folder_list):
    tokens=[]
    for folder in folder_list:
        print '遍历',folder,'目录下的文件'
        file_list = get_file_list(dir,folder)
        i=0
        for file in file_list:
            file_path = dir+'/'+folder+'/'+file
            f = open(file_path,'r')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha() or w.isdigit()]
            tokens += words
    return tokens

#主函数，获取训练集中的词典
def func1(trandir):
    
    train_folder_list  = get_folder_list(traindir)
    tokens = get_tokens(traindir,train_folder_list)
    tokens = set(tokens)
    print len(tokens)
    pickle.dump(tokens,open('./data/tokens','wb'))

#构建训练集，采用向量空间模型
def func2(traindir,tokens):
    folder_list = get_folder_list(traindir)
    train_x,train_y = [],[]
    category = {}
    i = 0
    length = len(tokens)
    for folder in folder_list:
        print('遍历%s目录下的文件'%(folder))
        category[folder] = i
        file_list = get_file_list(traindir,folder)
        for file in file_list:
            #建立一个list,前部分表示向量，最后一个表示类别
            temp = [0 for j in range(length+2)]
            file_path = traindir+'/'+folder+'/'+file
            f = open(file_path,'r')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha() or w.isdigit()]
            word_count =dict(nltk.FreqDist(words))
            for word in words:
                loc = tokens.index(word)
                temp[loc] = word_count[word]
            train_x.append(temp)
            train_y.append(i)
        i += 1
    return (train_x,train_y,category)

#构建测试集
def func3(testdir,tokens,category):
    folder_list = get_folder_list(testdir)
    test_x,test_file = [],[]
    result = []
    length = len(tokens)
    for folder in folder_list:
        print('遍历%s目录下的文件'%(folder))
        des_category = category[folder]
        file_list = get_file_list(testdir,folder)
        for file in file_list:
            #建立一个list，用于保存文档id和真实类别
            temp = [0 for j in range(2)]
            temp[0] = file
            temp[1] = des_category
            result.append(temp)
            #建立一个list，用于保存测试集的向量
            temp1 = [0 for j in range(length+1)]
            test_file.append(file)
            file_path = testdir+'/'+folder+'/'+file
            f = open(file_path,'r')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha() or w.isdigit()] 
            word_count = dict(nltk.FreqDist(words))
            for word in words:
                #here i ingore the word only occur in test
                if word in tokens:
                    loc = tokens.index(word)
                    temp1[loc] = word_count[word]
            test_x.append(temp1)
    return (result,test_x,test_file)
            
           
        
        
        

if __name__=="__main__":
    traindir = './data/training'
    testdir = './data/test'
    
    #func1(traindir)

    tokens = list(read('tokens'))
    

    train_x,train_y,category=func2(traindir,tokens)
    train_x = np.array(train_x)
    print train_x.shape
    '''
    write(train,'train')
    write(category,'category')
    '''
    '''
    category = read('category')
    result,test = func3(testdir,tokens,category)
    write(result,'result')
    write(test,'tests')
    '''

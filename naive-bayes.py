#!usr/bin/env python
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
import csv

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

#构建训练集
def func2(traindir):
    folder_list = get_folder_list(traindir)
    train = []
    category = {}
    i = 0
    for folder in folder_list:
        print('遍历%s目录下的文件'%(folder))
        category[folder] = i
        file_list = get_file_list(traindir,folder)
        for file in file_list:
            file_path = traindir+'/'+folder+'/'+file
            f = open(file_path,'r')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha() or w.isdigit()]
            #词只按出现与否进行处理，不考虑出现多少次
            words = list(set(words))
            temp = []
            temp.append(words)
            temp.append(file)
            temp.append(i)
            train.append(temp)
        i += 1
    return (train,category)

#构建测试集
def func3(testdir,category):
    folder_list = get_folder_list(testdir)
    test = []
    result = []
    for folder in folder_list:
        print('遍历%s目录下的文件'%(folder))
        des_category = category[folder]
        file_list = get_file_list(testdir,folder)
        for file in file_list:
            temp = [0 for i in range(2)]
            temp[0] = file
            temp[1] = des_category
            result.append(temp)
            file_path = testdir+'/'+folder+'/'+file
            f = open(file_path,'r')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha() or w.isdigit()]
            #每个词无论出现多少次，都按出现一次处理
            words = list(set(words))
            temp1 = []
            temp1.append(words)
            temp1.append(file)
            temp1.append(i)
            test.append(temp1)
        i += 1
    return (result,test)
def preprocess():
    traindir = './data/training'
    testdir = './data/test'
    train,category = func2(traindir)
    write(train,'train_nb')
    write(category,'category_nb')
    result,test = func3(testdir,category)
    write(test,'test_nb')
    write(result,'result_nb')
                        
#统计每一种类别文档个数，及没一个类别中文档中的词汇量
def sta_count(train):
    train = np.array(train)
    sta = {}
    for i in range(10):
        sta[i] = {}
        train_category = train[train[0::,2]==i]
        sta[i]['count'] = len(train_category)
        tokens = []
        alltokens = 0
        for doc in train_category:
            alltokens += len(doc[0])
            tokens = set(tokens)|set(doc[0])
        sta[i]['words'] = len(tokens)
        sta[i]['all'] = alltokens
    return sta
#计算在某一类别下某一单词出现的次数
def cal_count(word,train):
    count = 0
    for doc in train:
        if word in doc[0]:
            count += 1
    return count
#计算在哪一类别的概率最大
def cal_max_category(words,train,sta):
    max_pro,max_category = 0,10
    train = np.array(train)
    for i in range(10):
        train_category = train[train[0::,2]==i]
        pro = sta[i]['count']
        for word in words:
            count = cal_count(word,train_category)
            #拉普拉斯平滑
            #这里采用的是贝努利模型
            '''
            在这里做一下改进，由于分母过大，很可能导致０的出现，所以对分母进行适当的缩放．
            '''
            pro *=(count+1)/float(300)*(sta[0]['count']+9)/float(sta[i]['count']+9)
        if pro > max_pro:
            max_pro = pro
            max_category = i
    print max_category
    return max_category

#用朴素贝叶斯模型进行统计
def cal(test,train,sta):
    predict = []
    for i in range(len(test)):
        print('第%d个结果预测'%(i))
        words = test[i][0]
        #计算此文档在哪一类别下的概率最大
        max_category = cal_max_category(words,train,sta)
        temp = [0 for j in range(2)]
        temp[0] = test[i][1]
        temp[1] = max_category
        predict.append(temp)
    return predict
#对结果进行统计
def sta_result(predict,category,result,path):
    result = np.array(result)
    predict = np.array(predict)
    evaluate = {} 
    a,b,c=0,0,0
    csvfile = file(path,'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['category','precision(%)','recall(%)','recall(%)'])
    for i in range(10):
        s = category[i]
        evaluate[s] = {} 
        predict_category = predict[predict[0::,1]==str(i),0]
        result_category = result[result[0::,1]==str(i),0]
        hit = len(set(predict_category)&set(result_category))
        precision = float('%.1f'%(hit/float(len(predict_category))*100))
        recall =  float('%.1f'%(hit/float(len(result_category))*100))
        F_1 = float('%.1f'%(2*precision*recall/(precision+recall)))
        a += hit
        b += len(predict_category)
        c += len(result_category)
        evaluate[s][0] = precision*100
        evaluate[s][1] = recall*100
        evaluate[s][2] = F_1
        writer.writerow([s,precision,recall,F_1])
        print('%s\t,%.1f\t,%.1f\t,%.1f\n'%(s,precision,recall,F_1))
    evaluate['all'] = {}
    p = float('%.1f'%(a/float(b)*100))
    r = evaluate['all'][1] =  float('%.1f'%(a/float(c)*100))
    f = evaluate['all'][2] =  2*p*r/(p+r)
    evaluate['all'][0] = p
    evaluate['all'][1] = r
    evaluate['all'][2] = f
    writer.writerow(['avearge',p,f,r])
    print('%s\n'%(evaluate['all'].values()))
    return evaluate
        
def convert(category):
    a = dict()
    for key,value in category.iteritems():
        a[value] = key
    return a
    
if __name__=="__main__":
    train = read('train_nb')
    test = read('test_nb')
    category = read('category_nb')
    #数字类别到字符串类别的转换
    category_convert = convert(category)
    result = read('result_nb')
    sta = sta_count(train)
    '''
    predict = cal(test,train,sta)
    write(predict,'predict')
    '''
    predict = read('predict')
    path = './data/bernoulli.csv'
    evaluate = sta_result(predict,category_convert,result,path)
    write(evaluate,'bernoulli_evaluate')
    

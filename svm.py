#!/usr/bin/env python
#coding:utf-8

'''
function:支持向量机
author:jwchen
date:2014-05-15
'''
import naive_bayes as nb
import os 



if __name__=="__main__":

    category_tokens =nb.read('mi_category')
    
    train = nb.read('train_nb_eventmodel_mi')
    for i in range(10):
        print category_tokens[i]
        print train[:3]
        break

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:45:55 2018

@author: xiong
"""
import numpy as np

import os
os.chdir()##change file directory to where weight_method lies
from weight_method import *
def YearlyReturn(r):
    l=len(r)
    total_ret=np.exp(np.sum(r))-1
    y_ret=pow(1+total_ret,240/l)-1
    return y_ret

def TotalReturn(r):
    return np.exp(np.sum(r))-1

def YearlyVol(r):
    R=np.exp(r)-1
    return R.std()*np.sqrt(240)

def TotalVol(r):
    R=np.exp(r)-1
    return R.std()*np.sqrt(len(r))

def ShapeRatio(r):
    return YearlyReturn(r)/YearlyVol(r)

def Drawdown(r):
    new_r=np.maximum.accumulate(np.exp(np.cumsum(r)))
    drawdown=(np.exp(np.cumsum(r))-new_r)/new_r
    return drawdown

def Turnover(weight):
    w=copy.deepcopy(weight)
    for i in range(1,len(w)):
        w.iloc[i,:]=weight.iloc[i,:]-weight.iloc[i-1,:]
    w.iloc[0,:]=np.nan
    
    w.where(w>0,0,inplace=True)
    return w.sum(axis=1).mean()
    

def DR(ret,wgt,covtype='sample'):
    sigma=robustcovest(ret,covtype)
    numerator=np.sum(np.sqrt(np.diag(sigma))*wgt)
    denominator=np.sqrt(np.dot(np.dot(wgt,sigma),wgt))
    return numerator/denominator

def CVaR(ret,wgt,beta):
    combine_ret=np.dot(ret,wgt)
    sur_ret=combine_ret[combine_ret<=np.percentile(combine_ret,(1-beta)*100)]
    return sur_ret.mean()

def RealizedVar(ret,wgt,covtype='sample'):
    sigma=robustcovest(ret,covtype)
    return np.dot(np.dot(wgt,sigma),wgt)
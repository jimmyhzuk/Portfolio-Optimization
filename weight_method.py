# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 08:46:23 2018

@author: xiong
"""
import pandas as pd
import numpy as np
import sklearn.covariance as skc
import scipy.optimize as sco
import cvxopt as cvx
from cvxopt import matrix

class PortOptimizer():
    def __ini__(self):
        self.pena=0
        self.bound=1
        self.covtype='sample' #'LedoitWolf' ‘MinDet’
    
    def fitpena(self,pena):
        self.pena = pena
    def fitbnd(self,bnd):
        self.bound = bnd
    def cvar_con(self,beta):
        self.beta=beta
    def fitdf(self,data):
        self.df=data
    
    def show_methods(self):
        print('MinVar')
        print('MaxDiver')
        print('MinCVaR')
        print('EqualWeight')
        print('InverseVol')
        print('RiskParity')
    def construct(self,method,covtype='sample'):

        if not (method in ['EqualWeight','InverseVol','RiskParity','MinVar','MaxDiver','MinCVaR']):
            print('unknown methods...')
            raise Warning
        else:
            if(method=='EqualWeight'):
                wgt = pd.Series(1,index = self.df.columns)
                wgt = wgt/np.nansum(wgt)
            elif(method=='InverseVol'):
                sigma=robustcovest(self.df,covtype)
                wgt = np.sqrt(np.diag(sigma))**(-1)
                wgt[np.isinf(wgt)] = np.nan
                wgt = wgt/np.nansum(wgt)  #notice! NaN problem
            elif(method=='MinVar'):
                wgt=qua_opti(robustcovest(self.df,covtype)*10**4,self.pena,self.bound) ## in case that the convariance matrix is too small
            elif(method=='RiskParity'):
                wgt=riskparitywgtfind(robustcovest(self.df,covtype)*10**4,self.bound)
            elif(method=='MaxDiver'):
                wgt=MaxDiverwgtfind(robustcovest(self.df,covtype), self.pena, self.bound)
            else:
                wgt=cvar_find(self.df,self.beta,self.bound)
        return pd.Series(wgt,index=self.df.columns)
                
def MaxDiverwgtfind(sigma,pena=0.,bound=1.):
    '''one way
    diversification=lambda x: -np.dot(np.sqrt(np.diag(sigma)),x).sum()/np.sqrt(np.dot(x.T,np.dot(sigma,x)))
    cons=({'type':'eq', 'fun':lambda x: np.nansum(x)-1})
    w_ini=np.repeat(1,np.shape(sigma)[0])
    w_ini=w_ini/np.nansum(w_ini)
    bnds=((0,bound),)*np.shape(sigma)[0]
    result=sco.minimize(diversification,w_ini,bounds=bnds,constraints=cons,options={'ftol':10**-8,'disp':True})
    '''
    var=np.diag(sigma)
    corr=sigma/np.sqrt(np.mat(var).T*np.mat(var))
    w=np.array(qua_opti(corr,pena,bound)).T
    w=w*(np.sqrt(var)**(-1))
    wgt=w/w.sum()
    return wgt.reshape(len(sigma))

def qua_opti(Q,pena=0.0,bound=1.):
    Q = np.matrix(Q)
    n = Q.shape[0]  ##row number
    Q=Q+np.eye(n)*pena ##add pena
    Q=2*matrix(Q)
    q=matrix(0.,(n,1))
    
    G=np.vstack([np.eye(n),-np.eye(n)])
    G=matrix(G,(2*n,n))
    h=matrix([bound for i in range(n)]+[0. for i in range(n)],(2*n,1))##bound can be improved
    
    A=matrix(1.,(1,n))
    b=matrix(1.)
    cvx.solvers.options['show_progress']=False
    cvx.solvers.options['reltol']=10**(-8)
    qp=cvx.solvers.qp(Q,q,G=G,h=h,A=A,b=b)

    return qp['x']

def riskparitywgtfind(sigma,bound=1.0):
    def riskparity(x):
            n=len(sigma)
            w=np.mat(x).T
            port_var=np.sqrt(w.T*np.mat(sigma)*w)
            port_vec=np.mat(np.repeat(port_var/n,n)).T
            diag=np.mat(np.diag(x)/port_var)
            partial=np.mat(sigma)*w
            return np.square(port_vec-diag*partial).sum()
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) -1})
    bnds = ((0, bound),)* sigma.shape[0]
    w_ini = np.repeat(1,np.shape(sigma)[0])
    w_ini = w_ini / sum(w_ini)
    res = sco.minimize(riskparity, w_ini, bounds=bnds,constraints=cons,options={'disp':True,'ftol':10**-10})
    return res['x']

def cvar_find(ret,beta,bound=1.0):
    q=np.shape(ret)[0]
    n=np.shape(ret)[1]
    m=1/(q*(1-beta))
    
    c=np.array([1]+[m for k in range(q)]+[0 for k in range(n)])
    
    A1=np.mat(np.eye(1+q+n))
    A2=np.hstack([np.mat(np.repeat(1.,q)).T,np.eye(q),np.mat(np.array(ret))])
    A_ub=-np.vstack([A1,A2])
    b_ub=np.mat(np.repeat(0,1+2*q+n)).T
    
    A_eq=np.mat([0 for k in range(1+q)]+[1 for k in range(n)])
    b_eq=np.mat(1.)
    
    bnds=((0,np.inf),)*(1+q)+((0,bound),)*n
    linpro=sco.linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bnds,options={'disp':True})

    return linpro['x'][(1+q):]

 
def robustcovest(df,covtype):
    if (covtype == 'sample'):
        return pd.DataFrame(np.cov(df,rowvar=False,ddof=1),index=df.columns,columns=df.columns)
    if (covtype == 'LedoitWolf'):      
        lw = skc.LedoitWolf()
        return pd.DataFrame(lw.fit(np.matrix(df)).covariance_ ,index = df.columns,columns = df.columns)
    if (covtype == 'MinDet'):
        return pd.DataFrame(skc.MinCovDet().fit(df).covariance_,index = df.columns,columns = df.columns)

###############################################################



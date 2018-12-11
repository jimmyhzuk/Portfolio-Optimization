## suppse you have a dataset called 'df', with columns being stock return and rows being date, and it's a 'good' dataset

import pandas as pd
import numpy as np
import os
os.chdir()##where weight_method lies
from weight_method import *

bnd=.2;pena=0.0;beta=0.8;cov='sample'
window = 240 ## used to compute covariance
method=['InverseVol','RiskParity','MinVar','MaxDiver','MinCVaR'，'EqualWeight']

date = df.index
name = df.columns

for mymethod in method:
    exp=str(mymethod)+' bound='+str(bnd)+'pena='+str(pena)+'beta='+str(beta)+cov
    wgt=pd.DataFrame(np.NAN,index=date,columns=name)
    print('start '+str(mymethod),end='\r')
    for i in np.arange(window,len(df)):
        ret=df.iloc[i-240:i,:]
        port=PortOptimizer()
        port.fitbnd(bnd)
        port.fitdf(ret)
        port.fitpena(pena)
        port.cvar_con(beta)
        wgt.iloc[i-1,:]=port.construct(mymethod,cov).round(6)
        print('Process{}%'.format(round((i-window)/(len(df)-window)*100,4),end='\r'))
    
    wgt.dropna(axis=0,how='all',inplace=True)
    wgt.fillna(0,inplace=True)
    wgt.to_csv(exp+'_wgt.csv',encoding='utf_8_sig')

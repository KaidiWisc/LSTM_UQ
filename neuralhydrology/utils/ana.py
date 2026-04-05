# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:22:27 2024

@author: Kaidi Peng
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
import matplotlib as mt
mt.rcParams["font.size"]=15

run_dir = Path("D:\!WISC_Res\LSTM\MyCase\default\\test_run_0806_23467960")
# eval_run(run_dir=run_dir, period="test")

#  read original
with open(run_dir / "test" / "model_epoch075" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

results.keys()

results['02443500']['1D']['xr']

# extract observations and simulations
qobs = results['02443500']['1D']['xr']['QObs(mm/d)_obs']
qsim = results['02443500']['1D']['xr']['QObs(mm/d)_sim']


import numpy as np
ensnum=np.shape(qsim)[-1]

fig, ax = plt.subplots(figsize=(10,3),dpi=300)
ax.plot(qobs['date'][1400:1600], qobs[1400:1600],c="r")
if ensnum>1:
    values=[]
    for ii in range(ensnum):
        ax.plot(qsim['date'][1400:1600], qsim[1400:1600,0,ii],c="k",alpha=0.1)
        # values.append(metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim[:,:,ii].isel(time_step=-1)))
    
else:
    ax.plot(qsim['date'][1400:1600], qsim[1400:1600],c="k")
    # values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))

ax.set_ylabel("Discharge (mm/d)")
ax.set_title(f"Test period - KGE {results['02443500']['1D']['KGE']:.3f}")

#%%

import os
import xarray as xr
from netCDF4 import Dataset,num2date
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
import pandas as pd
mt.rcParams["font.size"]=15

ensize=np.shape(qsim)[-1]
maskhit= (qobs>0) # lat-lon-time
prodis=np.nansum((qsim[:,0,:] <=qobs),1)/ensize  # lat-lon-time
prodis[maskhit==0]=np.nan

plt.figure(figsize=(5,5),dpi=300)
plt.hist(prodis,density=False,bins=50,cumulative=False)
plt.show()

res=np.unique(prodis[prodis>=0],return_counts=True)
catg=res[0]
cnts=res[1]
cumcnts=np.cumsum(cnts)
Tcnts=np.sum(cnts)

plt.figure(figsize=(5,5),dpi=300)
plt.scatter(catg,cumcnts/Tcnts)
plt.plot([0,1],[0,1],ls="--",c="k")
plt.xlabel("Obs Quantile in Ensembles")
plt.ylabel("CDF")
plt.grid()
plt.show()

#%%  reliability diagram
import pandas as pd
import os
import xarray as xr
from netCDF4 import Dataset,num2date
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
mt.rcParams["font.size"]=15
ensize=np.shape(qsim)[-1]

obsq=0.5
thre=np.quantile(qobs,obsq)

prodis=np.nansum((qsim[:,0,:] > thre),1)/ensize 
res=np.unique(prodis.data,return_counts=True)
catg=res[0]
cnts=res[1]

catg=np.arange(0,51)/50  # maintain the same sampling interval
intv=catg[1]-catg[0]
colist=[]
for ii in catg:
    maskg=((prodis>=ii) & (prodis<ii+intv) & (qobs.data[:,0]>=0))  # select effective obs
    stagg=qobs[maskg.flatten()]
    colist.append(np.nansum((stagg>thre))/len(stagg)) # poppppp
    

plt.figure(figsize=(5,5),dpi=300)
plt.scatter(catg,colist)
plt.plot([0,1],[0,1],ls="--",c="k")
plt.xlabel("Ensemble > %0.02f"%(thre))
plt.ylabel("obs > %0.02f"%(thre))
plt.grid()
plt.show()


#%%  interesting trial to draw KGE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mt
from fun_ens_eva import calc_CR, calc_CRPS_ens, calc_MAE, calc_RMSE,calc_KGE
mt.rcParams["font.size"]=15

obsqlist=[0.99,0.9,0.8,0.5,0] 
kgelist=np.ones((5,100))*-9999
ensize=np.shape(qsim)[-1]
    
ii=0
for obsq in obsqlist:
    
    obs=qobs.data.copy()

    maxobs=np.quantile(obs,obsq)
    obs[obs<maxobs]=np.nan  
    
    for jj in range(ensize):
        
        kgelist[ii,jj]=calc_KGE(qsim[:,0,jj].data,obs[:,0])
    
    ii+=1
        
kgelist[kgelist==-9999]=np.nan
kgelist=np.transpose(kgelist)

#%
# kgelist = np.delete(kgelist, 349, axis=0)
plt.figure(figsize=(5,5),dpi=300)
plt.boxplot(kgelist[:,:],positions=np.arange(5))  #positions=np.arange(5)
plt.xlabel("% obs")
plt.ylim(0,1)
plt.ylabel("KGE")
plt.xticks(np.arange(5),obsqlist)
plt.grid()
plt.show()
    
#%% get entropy sample
# much more stable 
from scipy.stats import norm, gaussian_kde, rankdata
qobs = results['02443500']['1D']['xr']['QObs(mm/d)_obs'].data
qsim = results['02443500']['1D']['xr']['QObs(mm/d)_sim'].data
ensnum=np.shape(qsim)[-1]
tlen=np.shape(qsim)[0]
H_all=np.zeros((tlen))

for tt in range(tlen):

    print("%d of %d"%(tt,tlen))
    
    Y=qsim[tt,0,:].flatten()
    Y=Y[np.isnan(Y)==False]
    Ygrid = np.linspace(np.min(Y)-0.2, np.max(Y), 100)
    fy = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    H_all[tt]=np.trapz(-fy*np.log(fy+1e-300), Ygrid)
    print(np.trapz(fy, Ygrid))
    
#%% scatter entropy
fig, ax = plt.subplots(figsize=(5,5),dpi=300)
mapp=ax.hist2d(qobs[:,0], H_all,bins=50,cmin=1,norm="log",cmap="viridis")
fig.colorbar(mapp[3], ax=ax)
plt.xlabel("Qobs [mm/day]")
plt.ylabel("Entropy")
plt.grid()
plt.show()


#%% get entropy from histogram
# bins affect entropy a lot...
qobs = results['02443500']['1D']['xr']['QObs(mm/d)_obs'].data
qsim = results['02443500']['1D']['xr']['QObs(mm/d)_sim'].data
ensnum=np.shape(qsim)[-1]
tlen=np.shape(qsim)[0]
H_all=np.zeros((tlen))

for tt in range(tlen):

    print("%d of %d"%(tt,tlen))
    
    Y=qsim[tt,0,:].flatten()
    Y=Y[np.isnan(Y)==False]
    hist, bins = np.histogram(Y, bins=100, density=True)
    bin_widths = bins[1:] - bins[:-1]
    H_all[tt]=np.sum(-hist*np.log(hist+1e-300))* bin_widths[0]
    
    print(np.sum(hist)* bin_widths[0])
    

from scipy.stats import entropy
# (discrete) distribution
# I would believe that these are continuous
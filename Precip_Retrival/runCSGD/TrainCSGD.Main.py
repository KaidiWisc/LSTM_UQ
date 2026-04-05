
import os
import numpy as np
from netCDF4 import Dataset,date2num,num2date
import matplotlib.pyplot as plt
from datetime import date,datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import sys
idd=sys.argv[1]

iddlst=np.array(["01144000","01532000","03083500","06799100","06814000","06820500","06934000","07184000",
        "07227100","08150000","08150800","09504420","14137000","14189000","14301000"])

arealst=np.array([690,215,1715,671,276,1760,3180,197,
         786,1854,215,233,264,1790,667 ] )

area=float(arealst[iddlst==idd][0])
print(area)

#%% model Training =========================================
linear=False

imerg=pd.read_csv(f"../../CAMELS_US_ori/hourly/nldas_hourly/{idd}_hourly_img.csv")
img= imerg['apcp(mm/hr)']
imerg_=np.array(img)


pars=np.zeros(10)
Ndata = pd.read_csv("Input.txt", delimiter=' ',header=None)

pars[0] = np.array(Ndata[1][Ndata[0]=="clim_mu"])[0]
pars[1] = np.array(Ndata[1][Ndata[0]=="clim_sigma"])[0]
pars[2] = np.array(Ndata[1][Ndata[0]=="clim_delta"])[0]

pars[3] = np.array(Ndata[1][Ndata[0]=="alpha1"])[0]
pars[4] = np.array(Ndata[1][Ndata[0]=="alpha2"])[0]
pars[5] = np.array(Ndata[1][Ndata[0]=="alpha3"])[0]
pars[6] = np.array(Ndata[1][Ndata[0]=="alpha4"])[0]
pars[7] = np.array(Ndata[1][Ndata[0]=="alpha5"])[0]

pars[8]=np.nanmean(imerg_)
pars[9] =0


if linear==False:
    logarg=pars[4]+pars[5]*imerg_/pars[8]
    mu=pars[0]/pars[3]*np.log1p(np.expm1(pars[3])*logarg)
if linear ==True:
    arg=pars[4]+pars[5]*imerg_/pars[8]
    mu=pars[0]*arg
    
sigma=pars[6]*pars[1]*np.sqrt(mu/pars[0])
delta= pars[2] 

#%%
import scipy as sp
q99=delta+sp.stats.gamma.ppf(0.99,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q90=delta+sp.stats.gamma.ppf(0.90,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q75=delta+sp.stats.gamma.ppf(0.75,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q50=delta+sp.stats.gamma.ppf(0.50,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q25=delta+sp.stats.gamma.ppf(0.25,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q10=delta+sp.stats.gamma.ppf(0.10,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
q01=delta+sp.stats.gamma.ppf(0.01,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)

csgd=imerg.copy()
csgd['q99(mm/h)']=q99
csgd['q90(mm/h)']=q90
csgd['q75(mm/h)']=q75
csgd['q50(mm/h)']=q50
csgd['q25(mm/h)']=q25
csgd['q10(mm/h)']=q10
csgd['q01(mm/h)']=q01
csgd['date'] = pd.to_datetime(csgd['date'])
csgd = csgd.dropna(subset=['date'])
csgd = csgd.sort_values('date')
csgd = csgd.set_index('date') 

csgd.to_csv(f"../runLSTM/CAMELS_US/hourly/nldas_hourly/{idd}_hourly_nldas.csv",
             index=True,float_format='%.4f',na_rep=-999)

#%%

csgd_day = csgd.resample('D').mean() 
csgd_day['Year'] = csgd_day.index.year
csgd_day['Mnth'] = csgd_day.index.month
csgd_day['Day'] = csgd_day.index.day
csgd_day['Hr'] = csgd_day.index.hour


with open(f"../runLSTM/CAMELS_US/basin_mean_forcing/nldas/{idd}_lump_nldas_forcing_leap.txt", 'w') as f:
    f.writelines("43.7142 \n")  #latitude of gaugeX
    f.writelines("-72.4181 \n")  #  elevation of gauge (m)X
    f.writelines(f"{area*2.59*10**6} \n")  #area of basin (m^2)

    csgd_day.to_csv(f, sep='\t', index=False,lineterminator='\n',float_format='%.4f')

print("finish the create of new forcing")
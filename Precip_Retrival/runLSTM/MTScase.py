
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import os
import sys
idd=sys.argv[1]

iddlst=np.array(["01144000","01532000","03083500","06799100","06814000","06820500","06934000","07184000",
        "07227100","08150000","08150800","09504420","14137000","14189000","14301000"])

arealst=np.array([690,215,1715,671,276,1760,3180,197,
         786,1854,215,233,264,1790,667 ])

area=float(arealst[iddlst==idd][0])
print(area)

#%%
import sys
import shutil
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

def calc_KGE(predicted0, reference0, sr=1.0, salpha=1.0, sbeta=1.0):

    predicted=predicted0[(predicted0>=0) & (reference0>=0)]
    reference=reference0[(predicted0>=0) & (reference0>=0)]
    
    pdims= predicted.shape
    rdims= reference.shape

    std_ref = np.std(reference)
    if std_ref == 0:
        return -np.inf
    sum_ref = np.sum(reference)
    if sum_ref == 0:
        return -np.inf
    alpha = np.std(predicted) / std_ref
    beta = np.sum(predicted) / sum_ref
    cc = np.corrcoef(reference, predicted)[0, 1]

    # Calculate the kge09
    kge09 = 1.0 - np.sqrt((sr*(cc-1.0))**2 +
                          (salpha*(alpha-1.0))**2 +
                          (sbeta*(beta-1.0))**2)

    return kge09


# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    start_run(config_file=Path("MTS_basin.yml"))

# fall back to CPU-only mode
else:
    start_run(config_file=Path("MTS_basin.yml"), gpu=-1)


dirs=os.listdir("runs")
run_dir = Path("runs/%s/"%dirs[0])  # you'll find this path in the output of the training above.
run_config = Config(run_dir / "config.yml")


tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="train", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="validation", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)

rudir="runs/%s/"%dirs[0]
print(rudir)
os.mkdir(rudir+"models/")
for filename in os.listdir(rudir):
    if filename.endswith('.pt'):
        print(filename)
        file_path = os.path.join(rudir, filename)
        shutil.move(file_path, rudir+"models/")

shutil.move(rudir+"models/model_epoch100.pt",rudir)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="train", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="validation", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)


#--------------------------------------------------------------------------------------
kge1=[]

with open(rudir+"/test/model_epoch100/test_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim1 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum1=np.shape(qsim1)[0]
qsim1[qsim1<0]=0

for jj in range(0,ensnum1):    
    print(jj)
    kge1.append(calc_KGE(qsim1[jj,:],qobs))


with open(rudir+"/test/model_epoch200/test_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim2 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum2=np.shape(qsim2)[0]
qsim2[qsim2<0]=0

for jj in range(0,ensnum2):    
    print(jj)
    kge1.append(calc_KGE(qsim2[jj,:],qobs))

# pla=np.argmax(np.array(kge1))
medv = np.nanmedian(np.array(kge1))
pla = np.argmin(np.abs(np.array(kge1)-medv))
 
if pla<ensnum1:  #0-(ens1-1)
    bsim=qsim1[pla,:]
if pla>=ensnum1:
    bsim=qsim2[pla-ensnum1,:]

bsim_m3s = bsim / 1000 * (area*2.59*10**6)  / 60**2
# mm/h -> m/h * (mi2->m2) -> m3/h -> m3/s


data1 = {
    "sim[m3/s]": bsim_m3s,  # SIM values
    "dates": hourTime # Dates
}

import glob
files = glob.glob('Input*.txt')[0]
fnm = files[5:-4]

df = pd.DataFrame(data1)
df.to_csv(f"test_streamflow_sim_{fnm}.csv", sep=",", index=False)
print("finish the create of new streamflow")


#--------------------------------------------------------------------------------------
kge1=[]

with open(rudir+"/train/model_epoch100/train_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim1 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum1=np.shape(qsim1)[0]
qsim1[qsim1<0]=0

for jj in range(0,ensnum1):    
    print(jj)
    kge1.append(calc_KGE(qsim1[jj,:],qobs))


with open(rudir+"/train/model_epoch200/train_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim2 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum2=np.shape(qsim2)[0]
qsim2[qsim2<0]=0

for jj in range(0,ensnum2):    
    print(jj)
    kge1.append(calc_KGE(qsim2[jj,:],qobs))

medv = np.nanmedian(np.array(kge1))
pla = np.argmin(np.abs(np.array(kge1)-medv))
if pla<ensnum1:  #0-(ens1-1)
    bsim=qsim1[pla,:]
if pla>=ensnum1:
    bsim=qsim2[pla-ensnum1,:]

bsim_m3s = bsim / 1000 * (area*2.59*10**6)  / 60**2
# mm/h -> m/h -> m3/h -> m3/s


data1 = {
    "sim[m3/s]": bsim_m3s,  # SIM values
    "dates": hourTime # Dates
}

import glob
files = glob.glob('Input*.txt')[0]
fnm = files[5:-4]

df = pd.DataFrame(data1)
df.to_csv(f"train_streamflow_sim_{fnm}.csv", sep=",", index=False)
print("finish the create of new streamflow")

#--------------------------------------------------------------------------------------
kge1=[]

with open(rudir+"/validation/model_epoch100/validation_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim1 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum1=np.shape(qsim1)[0]
qsim1[qsim1<0]=0

for jj in range(0,ensnum1):    
    print(jj)
    kge1.append(calc_KGE(qsim1[jj,:],qobs))


with open(rudir+"/validation/model_epoch200/validation_results.p", "rb") as fp:
    results = pickle.load(fp)

hourly_xr = results[idd]["1H"]["xr"]
hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
hourTime= hourly_xr.coords['date'].data + np.array([np.timedelta64(h, 'h') for h in hourly_xr.coords['time_step'].data])
qobs = hourly_xr["QObs(mm/h)_obs"].data[:]
qsim2 = hourly_xr["QObs(mm/h)_sim"].data # ens, t           
ensnum2=np.shape(qsim2)[0]
qsim2[qsim2<0]=0

for jj in range(0,ensnum2):    
    print(jj)
    kge1.append(calc_KGE(qsim2[jj,:],qobs))

medv = np.nanmedian(np.array(kge1))
pla = np.argmin(np.abs(np.array(kge1)-medv))
if pla<ensnum1:  #0-(ens1-1)
    bsim=qsim1[pla,:]
if pla>=ensnum1:
    bsim=qsim2[pla-ensnum1,:]

bsim_m3s = bsim / 1000 * (area*2.59*10**6)  / 60**2
# mm/h -> m/h -> m3/h -> m3/s

data1 = {
    "sim[m3/s]": bsim_m3s,  # SIM values
    "dates": hourTime # Dates
}

import glob
files = glob.glob('Input*.txt')[0]
fnm = files[5:-4]

df = pd.DataFrame(data1)
df.to_csv(f"validation_streamflow_sim_{fnm}.csv", sep=",", index=False)
print("finish the create of new streamflow")

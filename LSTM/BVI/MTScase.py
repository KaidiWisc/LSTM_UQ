# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:02:42 2024

@author: Kaidi Peng
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import os

#%%
import sys
import shutil

# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    start_run(config_file=Path("MTS_basin.yml"))

# fall back to CPU-only mode
else:
    start_run(config_file=Path("MTS_basin.yml"), gpu=-1)


dirs=os.listdir("runs")
run_dir = Path("runs/%s/"%dirs[0])  # you'll find this path in the output of the training above.
run_config = Config(run_dir / "config.yml")

# create a tester instance and start evaluation
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="train", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)

rudir="runs/%s/"%dirs[0]
print(rudir)
os.mkdir(rudir+"models/")
for filename in os.listdir(rudir):
    if filename.endswith('.pt'):
        print(filename)
        file_path = os.path.join(rudir, filename)
        shutil.move(file_path, rudir+"models/")

import numpy as np
for ii in np.arange(5,501,5):
    shutil.move(rudir+f"models/model_epoch{ii:03d}.pt",rudir)
    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=True, metrics=run_config.metrics)
    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="train", init_model=True)
    results = tester.evaluate(save_results=True, metrics=run_config.metrics)
    shutil.move(rudir+f"model_epoch{ii:03d}.pt",rudir+"models")


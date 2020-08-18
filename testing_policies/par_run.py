import math
import networkx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random

import sys

sys.path.insert(1, '/covidtesting/models')
from models import *
from datetime import datetime
import json
import multiprocessing as mp
import pickle
import io
import os
import random

from multiprocessing import set_start_method


def make_graphs(numNodes = 1000, m=9, scale = 100 , plot_degree = False):
    baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=9)
    # Baseline normal interactions:
    G_normal     = custom_exponential_graph(baseGraph, scale=100)
    # Quarantine interactions:
    G_quarantine = custom_exponential_graph(baseGraph, scale=1) # changed to have extreme quarantine
    if plot_degree:
        plot_degree_distn(G_normal)
    return (G_normal,G_quarantine)


def split_params(p):
    exec_param_names = [ "T", "checkpoints", "print_interval", "verbose","runTillEnd"  , "stopping"  , "policyInterval" , "type", "variant" ] # , "policy"
    modelp = {}
    ep = {}
    for k in p:
        if k in exec_param_names:
            ep[k] = p[k]
        else:
            modelp[k] = p[k]
    return modelp, ep


def single_exec(T):
    x, keep = T
    modelp, ep  = split_params(x)
    if isinstance(modelp["G"], dict):
        G,Q = make_graphs(**modelp["G"])
        modelp["G"] = G
        modelp["Q"] = Q        
    to_print = False
    rid = random.randrange(1000)
    if random.random() < 0.001: # 
        to_print = True
        if "variant" in x:
            print(f"{rid:03d}: {x['variant']}")
        else:
            print(f"{rid:03d}")
    m = SEIRSNetworkModel(**modelp)
    row = m.run(**ep)
    if to_print:
        if "variant" in x:
            print(f"{rid:03d}: {x['variant']} -- DONE")
        else:
            print(f"{rid:03d} -- DONE")

    row["model"] = m if keep else None
    return row


def parallel_run(L, realizations=1):
    print("Preparing list to run", flush=True)
    run_list  = [(x,i<5) for i in range(realizations) for x in L]
    print(f"We have {mp.cpu_count()} CPUs")
    pool = mp.Pool(mp.cpu_count())
    
    print(f"Starting execution of {len(run_list)} runs", flush =True)
    #rows = list(map(single_exec, run_list))
    rows = list(pool.map(single_exec, run_list))
    print("done")
    pool.close()
    df =  pd.DataFrame(rows)
    return df



print("Loading to_run", flush=True)
with open('to_run.pickle', 'rb') as handle:
    to_run  = pickle.load(handle)
print("Loaded", flush = True)    


realizations = int(sys.argv[1])

#set_start_method("spawn")
data =  parallel_run(to_run,realizations)

print("Saving csv", flush = True)
data.to_csv('data.csv')

chunk_size = 100000
if data.shape[0] > chunk_size:
    print("Saving split parts",flush=True)
    i = 0
    for start in range(0,data.shape[0] , chunk_size):
        print(f"Saving pickle {i}")
        temp = data.iloc[start:start + chunk_size]
        temp.to_pickle(f'data_{i}.zip')
        i += 1
    print("Done", flush=True)
        

print("Saving overall pickle", flush = True)
data.to_pickle('data.zip')

print("Done")    

print(f"Size of data.csv is {os.path.getsize('data.csv')/1000:,.0f}K")
print(f"Size of data.zip  is {os.path.getsize('data.zip')/1000:,.0f}K")




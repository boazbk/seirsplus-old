import math
import networkx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random

from models import *
from datetime import datetime



def make_graphs(numNodes = 1000, m=9, scale = 100 , plot_degree = True):
    baseGraph    = networkx.barabasi_albert_graph(n=numNodes, m=m)
    # Baseline normal interactions:
    G_normal     = custom_exponential_graph(baseGraph, scale=scale)
    # Quarantine interactions:
    G_quarantine = custom_exponential_graph(baseGraph, scale=1) # changed to have extreme quarantine
    if plot_degree:
        plot_degree_distn(G_normal)
    return (G_normal,G_quarantine)

def split_params(p):
    exec_param_names = [ "T", "checkpoints", "print_interval", "verbose","runTillEnd"  , "stopping"  , "policyInterval" , "type", "variant" ] # , "policy"
    mp = {}
    ep = {}
    for k in p:
        if k in exec_param_names:
            ep[k] = p[k]
        else:
            mp[k] = p[k]
    return mp, ep


def run(params, type = None, realizations=1, plot= 1 , lview = None, verbose = False):
    def update_type(e):
        if type:
            e["type"] = type

    if not isinstance(params,list):
        params = [params]
    
    models = []
    for i in range(realizations):
        for x in params:
            mp, ep = split_params(x)
            update_type(ep)
            m = SEIRSNetworkModel(**mp)
            models.append( (m,ep   ))

    def f(T):
        m, p  = T
        if verbose:
            print(f"Running {p['variant']}:{p['type']}")
        row = m.run(**p)
        row["model"] = m
        return row
    print(f"Starting execution of {len(models)} runs")
    if lview:
        rows = list(lview.map(f, models))
    else:
        rows = list(map(f,models))
    print("Done")
    df =  pd.DataFrame(rows)
    if plot:
        plot_figures(df,num=plot)
    return df


basecolumns = ["excessRisk", "meanUndetected1st", "time1st", "meanUndetected", "meanTests"]

col_labels = {"t" : "Total days",
           "totUndetected" : "Total undetected person days",
           "totInfected": "Total infected person days",
           "finInfected": "No. Infected people at end",
           "totTests": "Total tests",
           "totPositive": "Total positive",
            "maxInfected": "Maximum Infected",
            "totUndetected1st" : "Total undetected person days until 1st",
            "totTests1st" : "Tests until first detection",
            "finS" : "Final susceptible",
            "meanTests" : "Avg tests per day",
            "meanUndetected": "Avg undetected per day",
            "meanUndetected1st" : "Avg undetected/day till detection",
            "meanTests1st": "Avg tests/day till detection",
            "time1st" : "Time 1st detection",
            "overall_infected" : "Overall infected throughout",
            "infected1st" : "Number infected at time of 1st detection",
            "excessRisk"  : "Excess risk over baseline (percentage)",
            "meanUndetectedInfectiousDays" : "Average number of undetected infectious persons per day"
            }



def plot_hists(df, types=None, columns=basecolumns , prefix =""):
    if columns == None or columns == "all":
        columns = col_labels.keys()
    if isinstance(columns, str):
        columns = [columns]

    types_ = list(df["type"].unique())
    if types == None:
        types = types_
    elif isinstance(types, str):
        types = [types]

    for c in columns:
        print(f"Plotting {col_labels[c]}:")
        fig = plt.figure(figsize=(20, 10), edgecolor="b")
        ax = fig.add_subplot(111)
        L = []
        for t in types:
            L.append(df[df["type"] == t][c].fillna(0).to_numpy(dtype=float))
        A = np.transpose(np.vstack(L))
        colors = sns.color_palette("hls", len(L))
        ax.hist(A, 10, density=True, histtype='bar', color=colors, label=types)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        if prefix != '':
            prefix = f" ({prefix})"
        ax.set_title(f'{col_labels[c]}{prefix}')
        # fig.text(.05, .00, f"{cols[c]}", ha='center')


def bot25(x): return np.percentile(x, 25)


def top25(x): return np.percentile(x, 75)


def summary(df, excel = False , csv = True, prefix ="", datadir = "."):
    types = list(df["type"].unique())
    columns = list(col_labels.keys())
    temp = df[["type"] + columns]
    temp = temp.rename(columns=col_labels)
    col_names = [col_labels[c] for c in columns]
    s =  pd.pivot_table(temp, index="type", aggfunc={np.mean, bot25, top25})
    s.columns = ['_'.join(col).strip() for col in s.columns]
    filename = f"{datadir}/summary_{prefix.replace('/','_')}"
    if excel:
        s.to_excel(filename+".xlsx", float_format="%.1f", encoding='utf8')
        print(f"Saved summary as {datadir}/{filename}.xlsx")
    if csv:
        s.to_csv(filename+".csv")
        print(f"Saved summary as {datadir}/{filename}.csv")
    return s





def plot_figures(df, types=None, num=1, ylim = 0.5):
    if types == None:
        types = list(df["type"].unique())
    elif isinstance(types, str):
        types = [types]
    for t in types:
        df = df[df["type"] == t]
        print(f"Plotting {num} sample executions of {t}")
        L = random.sample(range(len(df.index)), num)
        for a in L:
            df.iloc[a, :]["model"].figure_infections(plot_R="stacked", ylim=ylim)

            
            
from collections import defaultdict


def plot_batches(col,title= None, scale = 1, base_lines = ["No testing", "Business closed","Business closed (28 days)"], sumdata = None, prefix = "", ylabel = None, logscale = False , filename = None):
    if not title:
        title = col
    if not ylabel:
        ylabel = title
        
    plots = defaultdict(list)

    for type, row in sumdata.iterrows():
        i = type.find('/')
        if i < 0: continue
        if type.find(':')>0:
            type = type[type.find(':')+1:]
        days = int(type[:i])
        batches = int(type[i+1:])
        plots[days].append(
            ( batches,
              row[f"{col}_bot25"]*scale,
              row[f"{col}_mean"]*scale,
              row[f"{col}_top25"]*scale
            )
        )
    fig = plt.figure(figsize=(12, 6), dpi = 200)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Batch', fontsize = 16)
    if logscale:
        ax.set_yscale('log')
        label += " (log scale)"

    ax.set_ylabel(f'{ylabel}', fontsize = 16)
    
    

    colors = sns.color_palette("hls", len(list(plots.keys()))+len(base_lines))
    i = 0

    All_X = []

    zorder = 30

    for day in sorted(list(plots.keys())):
        X = []
        Y = []
        Ylower = []
        Yupper = []
        L = sorted(plots[day], key = lambda t: t[0])
        for a,b,c,d in L:
            X.append(a)
            Y.append(c)
            Ylower.append(b)
            Yupper.append(d)
        ax.plot(X,Y, color = colors[i], label = f"{day} days", zorder = zorder)    
        ax.fill_between(X,Ylower,Yupper, color = colors[i], alpha = 0.5, zorder = zorder-1)
        i += 1
        zorder -= 2
        All_X += X


    Base_X = sorted(list(set(All_X)))
    n = len(Base_X)
    for b in base_lines:
        low = sumdata.loc[b][f"{col}_bot25"]*scale
        mid = sumdata.loc[b][f"{col}_mean"]*scale
        top = sumdata.loc[b][f"{col}_top25"]*scale
        ax.plot(Base_X,[mid]*n, color= colors[i], label = f"{b}",linestyle="--")
        ax.fill_between(Base_X,[low]*n,[top]*n, alpha = 0.3, color = colors[i], zorder = zorder)
        zorder -= 1
        i += 1
    
    handles, labels = ax.get_legend_handles_labels()
    l = ax.legend([handle for i,handle in enumerate(handles)],  [label for i,label in enumerate(labels)], loc = 'best')
    l.set_zorder(50) 
    if prefix != '':
        prefix = f" ({prefix})"
    ax.set_title(f'{title} per batch size{prefix}')
    if filename:
        fig.savefig(filename)
    plt.show()


def scatter(col,title= None, scale = 1, sumdata = None , errorbars = False , prefix = "", filename = None):
    if not title:
        title = col
        
    plots = defaultdict(list)
    
    X = []
    Y = []
    labels = []
    Upper = []
    Lower = []
    colors = [] 

    color_palette  = sns.color_palette("Blues_r",57)
    
    for type, row in sumdata.iterrows():
        X += [row["Avg tests per day_mean"]]
        Y += [row[f"{col}_mean"]*scale]
        Upper += [ row[f"{col}_top25"]*scale - row[f"{col}_mean"]*scale ]
        Lower += [ row[f"{col}_mean"]*scale - row[f"{col}_bot25"]*scale ]
        labels += [ type ]
        if type.find("/")>0:
            c = int(type[type.find("/")+1:])
        else:
            c = 0
            
        colors += [ color_palette[c] ] 
        
    
    
    
    fig = plt.figure(figsize=(6, 6), dpi = 200)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Average number of tests/day', fontsize = 16)
    ax.set_ylabel(f'{title}', fontsize = 16)
    
    for i in range(len(X)):
        ax.scatter(X[i],Y[i],color = colors[i])
        ax.annotate(labels[i],(X[i],Y[i]), fontsize = "x-small")
    
    if errorbars:
        ax.errorbar(X,Y,yerr=[Lower,Upper], linestyle="None")
    if prefix != '':
        prefix = f" ({prefix})"

    ax.set_title(f'{title} vs tests{prefix}')
    if filename:
        fig.savefig(filename)
    plt.show()

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from new.SIR import DC,rank_list
from new.k_shell import kshell
from new.MDD import MDD
from new.Cnc_plus import Cnc_plus
from new.Ks_IF import Ks_IF
from new.H_index import H_index
from new.MCDE import MCSDWE
from new.TOPSIS import SUM_TOPSIS
def rank_list(G,mc,p):
    res = dict()
    for u in G.nodes():
        tmp = 0
        for i in range(mc):
            tmp += SIR(G,u,p)/mc
        res[u] = tmp
    #result = list(dict(sorted(res.items(), key=lambda x: x[1], reverse=True)).keys())
    return res
def DM(R):
    return len(set(R.values()))/len(R.keys())

def M(R):
    n = len(R)
    num_cnt = defaultdict(int)
    for x in R.values():
        num_cnt[x] += 1
    tmp = 0
    for k,v in num_cnt.items():
        if v > 1:
            tmp += (v*(v-1))/(n*(n-1))
    return (1 - tmp)**2

def Kendall_Tau(X,Y):
    n= len(X)
    XY = [(X[i],Y[i]) for i in range(len(X))]
    C = 0
    D = 0
    for i in range(n):
        for j in range(i,n):
            if XY[i][0] > XY[j][0] and XY[i][1] > XY[j][1] or \
                XY[i][0] < XY[j][0] and XY[i][1] < XY[j][1]:
                C += 1
            elif XY[i][0] > XY[j][0] and XY[i][1] < XY[j][1] or \
                XY[i][0] < XY[j][0] and XY[i][1] > XY[j][1]:
                D += 1
            else:
                continue
    return 2*(C-D)/(n*(n-1))

def CCDF(R):
    res = dict()
    n = len(R)
    infs = list(R.values())
    for i in range(1,n+1):
        max_inf = max(infs)
        tmp = 0
        for inf in infs:
            if inf <= max_inf:
                tmp += 1.0
        res[i] = tmp
        cnt = infs.count(max_inf)
        for j in range(cnt):
            infs.remove(max_inf)
        if len(infs) == 0:
            break
    for i in range(1,n+1):
        if i not in res.keys():
            res[i] = 0
    for i in range(1,n+1):
        res[i] = res[i]/n
    return res

def F(sir,stand):
    res = dict()
    X = list(dict(sorted(sir.items(), key=lambda x: x[1], reverse=True)).keys())
    Y = list(dict(sorted(stand.items(), key=lambda x: x[1], reverse=True)).keys())
    L = [5*i for i in range(1,21)]
    for num in L:
        res[num] = len(set(X[:num]).intersection(set(Y[:num])))/len(set(X[:num]).union(set(Y[:num])))
    return res

def J(XX,YY,L=5):
    X = list(dict(sorted(XX.items(), key=lambda x: x[1], reverse=True)).keys())
    Y = list(dict(sorted(YY.items(), key=lambda x: x[1], reverse=True)).keys())
    return len(set(X[:L]).intersection(set(Y[:L])))/len(set(X[:L]).union(set(Y[:L])))

#M
def f1():
    files = ['Karate','Contiguous','Dolphins','Copperfield','Jazz','Netscience','Elegans','Email','Euroroad','HAM','PowerGrid','PGP']
    for filename in files:
        print(filename)
        if filename == 'Elegans':
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
        with open('../dataSet/'+filename+'.txt') as f:
            for line in f:
                n = line.split()
                G.add_edge(n[0], n[1])
        for f in funcs:
            GG = G.copy()
            print(f.__name__,round(M(f.__call__(GG)),4),sep=" : ")
        print('--------------\n')

#Kendall
def f2():
    # files = ['Karate','Contiguous','Dolphins','Copperfield','Jazz','Netscience','Elegans','Email','Euroroad','HAM','PowerGrid','PGP']
    files = ['Elegans']
    # p = [0.15,0.25,0.15,0.1,0.05,0.15,0.01,0.1,0.35,0.03,0.3,0.1]
    p = [0.01]
    for i, filename in enumerate(files):
        print(filename)
        if filename == 'Elegans':
            G = nx.Graph()
        else:
            G = nx.Graph()
        with open('../dataSet/' + filename + '.txt') as f:
            for line in f:
                n = line.split()
                G.add_edge(n[0], n[1])
        # print('Nodes:',len(G.nodes()))
        # print('Edges:',len(G.edges()))
        if len(G) <= 100:
            mc = 10000
        elif len(G) > 100 and len(G) <= 10000:
            mc = 1000
        else:
            mc = 100
        sir = rank_list(G, mc, p[i])
        for f in funcs:
            GG = G.copy()
            ff = f.__call__(GG)
            X = []
            Y = []
            for u in G.nodes():
                X.append(ff[u])
                Y.append(sir[u])
            res = round(Kendall_Tau(X, Y), 4)
            print(f.__name__, res, sep=':')
        print('------------------------------')

#CCDF
def f3(G):
    for i,f in enumerate(funcs):
        GG = G.copy()
        R = f.__call__(GG)
        res = CCDF(R)
        value = list(res.keys())
        square = list(res.values())
        plt.plot(value, square, styles[i], linewidth=1,label=labels[i],color=colors[i],markerfacecolor=markerfacecolors[i],markersize='4',markevery=np.arange(1, 198,5))
    plt.xlim((1, 198+5)) #x坐标轴范围
    plt.xlabel("Rank", fontsize=9)  # 设置轴标题，并给定字号,设置颜色
    plt.ylabel("CCDF", fontsize=14)
    #plt.tight_layout()
    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, 198+5,5)
    plt.xticks(my_x_ticks,rotation=90)
    plt.tick_params(axis='both', labelsize=9)  # 设置刻度标记的大小
    plt.legend()
    plt.savefig('D:\\Jazz.tiff',dpi=500,bbox_inches = 'tight')
    plt.show()  #
#Jc
def f4():
    files = ['Karate', 'Contiguous', 'Dolphins', 'Copperfield', 'Jazz', 'Netscience', 'Elegans', 'Email', 'Euroroad',
             'HAM', 'PowerGrid', 'PGP']
    # files = ['Elegans']
    p = [0.15, 0.25, 0.15, 0.1, 0.05, 0.15, 0.01, 0.1, 0.35, 0.03, 0.3, 0.1]
    # p = [0.01]
    for i, filename in enumerate(files):
        print(filename)
        if filename == 'Elegans':
            G = nx.Graph()
        else:
            G = nx.Graph()
        with open('../dataSet/' + filename + '.txt') as f:
            for line in f:
                n = line.split()
                G.add_edge(n[0], n[1])
        # print('Nodes:',len(G.nodes()))
        # print('Edges:',len(G.edges()))
        if len(G) <= 100:
            mc = 10000
        elif len(G) > 100 and len(G) <= 10000:
            mc = 1000
        else:
            mc = 100
        sir = rank_list(G, mc, p[i])
        for i, f in enumerate(funcs):
            GG = G.copy()
            R = f.__call__(GG)
            res = F(sir, R)
            value = list(res.keys())
            square = list(res.values())
            plt.plot(value, square, styles[i], linewidth=1, label=labels[i], color=colors[i],
                     markerfacecolor=markerfacecolors[i], markersize='4')
        plt.title(filename)
        plt.xlim((5, 100))  # x坐标轴范围
        plt.xlabel("L", fontsize=9)  # 设置轴标题，并给定字号,设置颜色
        plt.ylabel("Jc", fontsize=14)
        # plt.tight_layout()
        # 设置坐标轴刻度
        my_x_ticks = np.arange(5, 100, 5)
        # plt.xticks(my_x_ticks, rotation=90)
        plt.xticks(my_x_ticks)
        plt.tick_params(axis='both', labelsize=9)  # 设置刻度标记的大小
        if filename == 'Jazz':
            plt.legend()
        plt.savefig('D:\\' + filename + '.tiff', dpi=500, bbox_inches='tight')
        plt.show()  #

def f5():
    G = nx.MultiGraph()
    with open('../dataSet/Dolphins.txt') as f:
        for line in f:
            n = line.split()
            G.add_edge(n[0], n[1])
    print('Nodes:', len(G.nodes()))
    if len(G) <= 100:
        mc = 10000
    elif len(G) > 100 and len(G) <= 10000:
        mc = 1000
    else:
        mc = 100

    for i, f in enumerate(funcs):
        ress = []
        GG = G.copy()
        R = f.__call__(GG)
        for p in pp:
            sir = rank_list(G, mc, p)
            X = []
            Y = []
            for u in G.nodes():
                X.append(R[u])
                Y.append(sir[u])
            res = round(Kendall_Tau(X, Y), 4)
            ress.append(res)
        plt.plot(pp, ress, styles[i], linewidth=1, label=labels[i], color=colors[i],
                 markerfacecolor=markerfacecolors[i],
                 markersize='4')
    plt.xlim((0, 0.2 + 0.01))  # x坐标轴范围
    plt.ylim((0.3, 0.9 + 0.1))  # x坐标轴范围
    plt.xlabel("β", fontsize=9)  # 设置轴标题，并给定字号,设置颜色
    plt.ylabel("τ", fontsize=14)
    # plt.tight_layout()
    # 设置坐标轴刻度
    my_x_ticks = np.arange(0, 0.2 + 0.01, 0.05)
    my_y_ticks = np.arange(0.3, 0.9 + 0.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.tick_params(axis='both', labelsize=9)  # 设置刻度标记的大小
    # plt.legend()
    plt.savefig('D:\\Dolphins.tiff', dpi=500, bbox_inches='tight')
    plt.show()  #


#funcs = [DC,kshell,MDD,Cnc_plus,Ks_IF,H_index,MCSDWE,SUM_TOPSIS]
funcs = [Ks_IF]
#funcs = [kshell]
styles = ['^-','s-','x-','D-','^-','o-','s-','o-']
colors=['black','red','yellow','green','orange','deepskyblue','purple','lightskyblue']
labels=['DC','Ks','MDD','Cnc+','Ks_IF','H-index','MCSDWE','SUM_TOPSIS']
markerfacecolors = ['black','red','yellow','none','none','deepskyblue','none','none']
pp = np.arange(0.01,0.2,0.01)
if __name__ == '__main__':
    print('单位：毫秒')
    #filenames = ['Karate','Contiguous','Dolphins','Copperfield','Jazz','Netscience','Elegans','Email','Euroroad','HAM','PowerGrid','PGP']
    filenames = ['PGP']
    for filename in filenames:
        print(filename)
        G = nx.MultiGraph()
        with open('../dataSet/' + filename + '.txt') as f:
            for line in f:
                n = line.split()
                G.add_edge(n[0], n[1])
        for f in funcs:
            start = time.time()
            for i in range(100):
                GG = G.copy()
                ff = f.__call__(GG)
            end = time.time()
            print(f.__name__,round(((end-start)/100)*1000,3),sep=':')
        print('---------------------\n')






















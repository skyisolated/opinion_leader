import networkx as nx
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
def F(sir,stand):
    res = dict()
    X = list(dict(sorted(sir.items(), key=lambda x: x[1], reverse=True)).keys())
    Y = list(dict(sorted(stand.items(), key=lambda x: x[1], reverse=True)).keys())
    L = [20*i for i in range(1,21)]
    for num in L:
        res[num] = len(set(X[:num]).intersection(set(Y[:num])))/len(set(X[:num]).union(set(Y[:num])))
    return res
def M(R):
    n = len(R)
    num_cnt = defaultdict(int)
    for x in R.values():
        num_cnt[x] += 1

    tmp = 0
    for k,v in num_cnt.items():
        if v > 1:
            tmp += (v*(v-1))/(n*(n-1))
    return round((1 - tmp)**2,4)
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
def probability(G):
    deg1 = sum(dict(G.degree).values()) / len(G)
    deg2 = dict()
    for v in G.nodes():
        nodes = []
        for u in G[v]:
            nodes += G[u]
        # nodes1 = set(nodes) - set(G[v]) - {v}
        nodes1 = set(nodes)
        deg2[v] = len(nodes)
    deg = sum(deg2.values()) / len(G)
    return round(deg1 / deg, 3)
def SIR(G,v,p):
    injected = [v]
    recovered = []
    while len(injected) > 0:
        for u in injected:
            for nb in G[u]:
                if nb in injected:
                    continue
                if nb in recovered:
                    continue
                if random.random() < p:
                    injected.append(nb)
            injected.remove(u)
            recovered.append(u)
    return len(recovered)
def rank_list(G,mc,p):
    res = dict()
    cnt = len(G)
    for u in G.nodes():
        tmp = 0
        for i in range(mc):
            tmp += SIR(G,u,p)/mc
        #print('Node {} over.'.format(u))
        res[u] = tmp
        cnt -= 1
        #print(cnt)
    #result = list(dict(sorted(res.items(), key=lambda x: x[1], reverse=True)).keys())
    return res
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
def nbs_degree(G,v):
    res = 0
    for u in G[v]:
        res += len(G[u])
    return res
def local_centrality(G):
    deg = dict(nx.degree(G))
    return deg
def global_centrality(G):
    GC = dict()
    for v in G.nodes():
        sum_deg = nbs_degree(G,v)
        GCv = 0
        for u in G[v]:
            GCv -= len(G[u])/sum_deg*np.log(len(G[u])/sum_deg)
        GC[v]  = GCv
    return GC
def SLD(G):
    K = dict(nx.degree(G))
    N = dict()
    for v in G.nodes():
        nodes = []
        nodes += G[v]
        for u in G[v]:
            nodes += G[u]
        N[v] = len(set(nodes)) - 1
    Q = dict()
    for v in G.nodes():
        tmp = 0
        for u in G[v]:
            tmp += N[u]
        Q[v] = tmp
    C = dict()
    for v in G.nodes():
        tmp1 = 0
        for u in G[v]:
            tmp1 += Q[u]
        C[v] = tmp1
    return C
def hIndex(citations):
    citations.sort()
    l = len(citations)
    lo, hi = 0, l - 1
    res = 0
    while (lo <= hi):
        mid = lo + (hi - lo) // 2
        cnt = l - mid  # 包括mid自身右边还有的元素个数
        if citations[mid] >= cnt:
            res = cnt
            hi = mid - 1
        else:
            lo = mid + 1
    return res
def H_index(G):
    H = dict()
    deg = G.degree()
    for v in G.nodes():
        deg1 = [ deg[u] for u in G[v] ]
        H[v] = hIndex(deg1)
    return H
def DC(G):
    return dict(nx.degree(G))
def CC(G):
    return nx.closeness_centrality(G)
def BC(G):
    return nx.betweenness_centrality(G)
def EC(G):
    return nx.eigenvector_centrality(G)
def PR(G):
    return nx.pagerank(G)
def HX(G):
    return H_index(G)
def KS(G):
    GG = G.copy()
    GG.remove_edges_from(nx.selfloop_edges(GG))
    return nx.core_number(GG)
def NEC(G):
    res = dict()
    deg = dict(nx.degree(G))
    for v in G.nodes():
        tmp = 0
        for u in G[v]:
            sum = nbs_degree(G,v)
            tmp -= (deg[u]/sum)*(np.log(deg[u]/sum))
        res[v] = tmp
    return res
def ENEC(G):
    nec = NEC(G)
    res = dict()
    for v in G.nodes():
        tmp = 0
        for u in G[v]:
            tmp += nec[u]
        res[v] = tmp
    return res
def SNEC(G):
    deg = dict(nx.degree(G))
    res = dict()
    for v in G.nodes():
        nodes = G[v]
        for u in G[v]:
            nodes = []
            nodes += G[u]
        nodes = set(nodes) - set(v)
        sum_deg = 0
        for nb in nodes:
            sum_deg += deg[nb]
        tmp = 0
        for u in nodes:
            tmp -= (deg[u] / sum_deg) * (np.log(deg[u] / sum_deg))
        res[v] = tmp
    res1 = dict()
    for v in G.nodes():
        tmp1 = 0
        for u in G[v]:
            tmp1 += res[u]
        res1[v] = tmp1
    return res
def ESNEC(G):
    snec = SNEC(G)
    res = dict()
    for v in G.nodes():
        tmp = 0
        for u in G[v]:
            tmp += snec[u]
        res[v] = tmp
    return res
def LGC(G):  #度X一阶熵
    res = dict()
    LC = local_centrality(G)
    GC = global_centrality(G)
    for v in G.nodes():
        res[v] = LC[v]*GC[v]
    res1 = defaultdict(int)
    for v in G.nodes():
        for u in G[v]:
            res1[v] += res[u]
    return res1
def ELGC(G):
    res = dict()
    deg = dict(nx.degree(G))
    g = global_centrality(G)
    l = dict()
    for v in G.nodes():
        nbs_deg = nbs_degree(G,v)
        l[v] = deg[v] / nbs_deg
    for v in G.nodes():
        res[v] = l[v]*g[v]
    res1 = dict()
    for v in G.nodes():
        tmp = 0
        for u in G[v]:
            tmp += res[u]
        res1[v] = tmp
    return res1

if __name__ == '__main__':
    x = np.arange(6)
    y = [4.828, 4.926, 0.012, 0.019, 0.031,0.234]
    y1 = [2, 6, 3, 8, 5,1]
    bar_width = 0.06
    tick_label = ["Email", "PowerGrid", "GLA", "AST", "Astro","TWT"]
    plt.bar(x, y, align="center", color="c", width=bar_width, label="title_A", alpha=0.5)
    #plt.bar(x + bar_width, y1, align="center", color="b", width=bar_width, label="title_B", alpha=0.5)
    plt.xticks(x, tick_label)
    plt.legend()
    plt.show()

    #files = ['Karate','Contiguous','Dolphins','Copperfield','Jazz','USAir97','Netscience','Elegans',
    #          'Email','Euroroad','HAM','PowerGrid','GRQC','GLA','PGP','Astro','AST','TWT','Enron']
    #p = [0.15, 0.25, 0.15, 0.1, 0.05, 0.05, 0.15, 0.01, 0.1, 0.35, 0.03, 0.3,
    #     0.1, 0.1, 0.1, 0.03, 0.015, 0.03, 0.01]
    # funcs = [BC, CC, DC, HX, KS, ENEC]
    # styles = ['p-', 's-', 'x-', 'D-', '^-', 'o-']
    # colors = [ 'purple', 'yellow', 'green', 'orange', 'deepskyblue', 'red']
    # labels = ['BC', 'CC', 'DC', 'HX', 'KS', 'NEC']
    # markerfacecolors = ['purple', 'yellow', 'green', 'orange', 'deepskyblue', 'red']
    # G = nx.Graph()
    # filename = 'TWT'
    # with open('data/'+filename+'.txt') as f:
    #     for line in f:
    #         n = line.split()
    #         G.add_edge(n[0], n[1])
    # print('Nodes:',len(G.nodes()))
    # print('Edges:',len(G.edges()))
    # print(nx.average_clustering(G))
        # M函数
        # R = ELGC(G)
        # print(filename, round(M(R),4),sep=' ------- ')

        #Kendall
        # sir = rank_list(G, 10, p[i])
        # with open('real/'+filename+'.txt','a') as ff:
        #     ff.write(str(sir))
        # ff = open('real/'+filename+'.txt', 'r')
        # data = ff.read()
        # sir = eval(data)
        # for f in funcs:
        #     GG = G.copy()
        #     ff = f.__call__(GG)
        #     X = []
        #     Y = []
        #     for u in G.nodes():
        #         X.append(ff[u])
        #         Y.append(sir[u])
        #     res = round(Kendall_Tau(X, Y), 4)
        #     print(f.__name__, res, sep='  ---------  ')
        # print('------------------------------')
    # for i, f in enumerate(funcs):
    #     GG = G.copy()
    #     R = f.__call__(GG)
    #     res = CCDF(R)
    #     value = list(res.keys())
    #     square = list(res.values())
    #     plt.plot(value, square, styles[i], linewidth=1, label=labels[i], color=colors[i],
    #              markerfacecolor=markerfacecolors[i], markersize='4',markevery=list(np.arange(1, end, step)))
    # plt.xlim((1, end+step))  # x坐标轴范围
    # plt.xlabel("Rank", fontsize=9)  # 设置轴标题，并给定字号,设置颜色
    # plt.ylabel("CCDF", fontsize=14)
    # # plt.tight_layout()
    # # 设置坐标轴刻度
    # my_x_ticks = np.arange(1, end+step, step)
    # plt.xticks(my_x_ticks, rotation=90)
    # plt.tick_params(axis='both', labelsize=9)  # 设置刻度标记的大小
    # plt.legend()
    # #plt.savefig('D:\\'+filename+'.tiff', dpi=500, bbox_inches='tight')
    # plt.savefig('D:\\'+filename+'.tiff', dpi=500, bbox_inches='tight')
    # plt.show()  #

# Jaccard
    #ff = open('real/'+filename+'.txt', 'r')
    #data = ff.read()
    #sir = eval(data)
    # sir = rank_list(G, 1000, p)
    # for i, f in enumerate(funcs):
    #     GG = G.copy()
    #     R = f.__call__(GG)
    #     res = F(sir, R)
    #     value = list(res.keys())
    #     square = list(res.values())
    #     plt.plot(value, square, styles[i], linewidth=1, label=labels[i], color=colors[i],
    #              markerfacecolor=markerfacecolors[i], markersize='4')
    # plt.title(filename)
    # plt.xlim((1, end+step))  # x坐标轴范围
    # plt.ylim((0, 1.2))
    # plt.xlabel("L", fontsize=9)  # 设置轴标题，并给定字号,设置颜色
    # plt.ylabel("F(L)", fontsize=14)
    # # plt.tight_layout()
    # # 设置坐标轴刻度
    # my_x_ticks = np.arange(0, end+step, step)
    # my_y_ticks = np.arange(0, 1.1, 0.1)
    # # plt.xticks(my_x_ticks, rotation=90)
    # plt.xticks(my_x_ticks,rotation=90)
    # plt.yticks(my_y_ticks)
    # plt.tick_params(axis='both', labelsize=9)  # 设置刻度标记的大小
    # plt.legend(loc=4)
    # plt.savefig('D:\\' + filename + '.tiff', dpi=500, bbox_inches='tight')
    # plt.show()  #
    # print('单位：秒')
    # filenames = ['TWT']
    # funcs = [BC,CC,DC,HX,KS,NEC]
    # for filename in filenames:
    #     print(filename)
    #     G = nx.Graph()
    #     with open('data/' + filename + '.txt') as f:
    #         for line in f:
    #             n = line.split()
    #             G.add_edge(n[0], n[1])
    #     for f in funcs:
    #         start = time.time()
    #         GG = G.copy()
    #         ff = f.__call__(GG)
    #         end = time.time()
    #         print(f.__name__, round(((end - start)), 3), sep=' --- ')
    #     print('---------------------\n')
import numpy as np
import json
import networkx as nx
import random
from collections import defaultdict
def SNEC(G):
    deg = dict(nx.degree(G))
    res = dict()
    for v in G.nodes():
        nodes = []
        nodes += G[v]
        for u in G[v]:
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
    return res1


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
if __name__ == '__main__':
    # G = nx.Graph()
    # with open('data/PowerGrid.txt','r') as f:
    #     for line in f:
    #         n = line.split()
    #         G.add_edge(n[0],n[1])
    # print('Nodes:',len(G.nodes()))
    # print('Edges:',len(G.edges()))
    # data = SNEC(G)
    # txt = str(data)
    # with open('output/PowerGrid.txt','a') as f:
    #     f.write(txt)
    f = open('output/Email.txt','r')
    data = f.read()
    a = eval(data)
    print(M(a))



import os
import networkx as nx

data_folder = "./fb100/data/" 
files = [f for f in os.listdir(data_folder) if f.endswith('.gml')]

small_graphs = []
k = 0
for file in files:
    path = os.path.join(data_folder, file)
    k += 1
    print(k)

    G = nx.read_gml(path)
    n = G.number_of_nodes()

    if n<2000:
        small_graphs.append(path)
    if len(small_graphs) == 10:
        break

print(small_graphs)
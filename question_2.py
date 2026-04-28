import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

files = {
    "Caltech": "fb100/data/Caltech36.gml",
    "MIT": "fb100/data/MIT8.gml",
    "Johns Hopkins": "fb100/data/Johns Hopkins55.gml"
}

# a)

def plot_degree_distrib(files):
    results = {}

    for name, path in files.items():
        G = nx.read_gml(path)
        
        global_clustering = nx.transitivity(G)
        mean_local_clustering = nx.average_clustering(G)
        density = nx.density(G)
        
        results[name] = {
            "Global Clustering": global_clustering,
            "Mean Local Clustering": mean_local_clustering,
            "Density": density,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges()
        }

        degrees = [d for n, d in G.degree()]
        plt.figure(figsize=(8, 4))
        plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Degrees distribution - {name}")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

    for name, data in results.items():
        print(f"--- {name} ---")
        for k, v in data.items():
            print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")

plot_degree_distrib(files)

# b)

def plot_degree_vs_clustering(files):
    for name, path in files.items():
        G = nx.read_gml(path)
        
        degree_dict = dict(G.degree())
        clustering_dict = nx.clustering(G)
        
        nodes = list(degree_dict.keys())
        degrees = [degree_dict[n] for n in nodes]
        clustering_coeffs = [clustering_dict[n] for n in nodes]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(degrees, clustering_coeffs, alpha=0.5, s=10, color='teal')
        
        plt.title(f"Degree vs Local Clustering - {name}")
        plt.xlabel("Degree (k)")
        plt.ylabel("Local Clustering Coefficient (C)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

plot_degree_vs_clustering(files)
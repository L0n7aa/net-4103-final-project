import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Liste des fichiers (adapte les noms selon tes fichiers réels)
files = {
    "Caltech": "fb100/data/Caltech36.gml",
    "MIT": "fb100/data/MIT8.gml",
    "Johns Hopkins": "fb100/data/Johns Hopkins55.gml"
}

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

    # 4. Tracé de la distribution des degrés [cite: 99]
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 4))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Degrees distribution - {name}")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

# Affichage des résultats pour l'analyse
for name, data in results.items():
    print(f"--- {name} ---")
    for k, v in data.items():
        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")

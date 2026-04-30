import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

data_folder = "./fb100/data/" 
files = [f for f in os.listdir(data_folder) if f.endswith('.gml')]

all_results = []

print(f"Traitement de {len(files)} réseaux...")
k = 0
for file in files:
    path = os.path.join(data_folder, file)
    G = nx.read_gml(path)
    k += 1
    print(k)
    
    n = G.number_of_nodes()
    
    try:
        r_status = nx.attribute_assortativity_coefficient(G, 'student_fac')
        r_major = nx.attribute_assortativity_coefficient(G, 'major_index')
        r_dorm = nx.attribute_assortativity_coefficient(G, 'dorm')
        r_gender = nx.attribute_assortativity_coefficient(G, 'gender')
        r_degree = nx.degree_assortativity_coefficient(G)
        
        all_results.append({
            'name': file,
            'n': n,
            'status': r_status,
            'major': r_major,
            'dorm': r_dorm,
            'gender': r_gender,
            'degree': r_degree
        })
    except KeyError as e:
        print(f"Attribut manquant dans {file}: {e}")

# Conversion en DataFrame pour manipuler les données facilement
df = pd.DataFrame(all_results)

attributes = ['status', 'major', 'dorm', 'gender', 'degree']

# Scatter Plots 
plt.figure(figsize=(15, 10))
for i, attr in enumerate(attributes, 1):
    plt.subplot(2, 3, i)
    plt.scatter(df['n'], df[attr], alpha=0.6, s=20)
    plt.title(f"Assortativité : {attr}")
    plt.xlabel("Taille du réseau (n)")
    plt.ylabel("Coefficient r")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Histogrammes
plt.figure(figsize=(15, 10))
for i, attr in enumerate(attributes, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[attr].dropna(), bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution : {attr}")
    plt.xlabel("Coefficient r")
    plt.ylabel("Fréquence")

plt.tight_layout()
plt.show()
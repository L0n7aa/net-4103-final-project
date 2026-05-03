import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

# 1. Définition de l'Encodeur GCN
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Première couche avec activation ReLU
        x = self.conv1(x, edge_index).relu()
        # Seconde couche qui génère les embeddings finaux (Z)
        return self.conv2(x, edge_index)
    
def evaluate_gnn_top_k(z, train_data, test_data, k_list):
    """Calcule Precision@k et Recall@k à partir des embeddings du GNN"""
    # 1. Calculer les probabilités
    adj_pred = torch.sigmoid(torch.matmul(z, z.t())).cpu().numpy()
    
    # 2. Masquer les arêtes déjà vues durant l'entraînement
    # On utilise edge_index car c'est là que sont stockées les arêtes d'entraînement
    edge_index_train = train_data.edge_index.cpu().numpy()
    adj_pred[edge_index_train[0], edge_index_train[1]] = 0
    adj_pred[edge_index_train[1], edge_index_train[0]] = 0
    np.fill_diagonal(adj_pred, 0)
    
    # 3. Trier les probabilités
    flat_adj = adj_pred.flatten()
    top_indices = np.argsort(flat_adj)[::-1]
    
    # 4. Identifier SEULEMENT les arêtes de test positives (les vrais liens cachés)
    # Dans test_data, les vraies arêtes sont celles où edge_label == 1
    mask_pos = (test_data.edge_label == 1)
    pos_edges = test_data.edge_label_index[:, mask_pos].cpu().numpy()
    true_edges_set = set(zip(pos_edges[0], pos_edges[1]))
    num_removed = len(true_edges_set)
    
    n_nodes = z.size(0)
    results = {}
    
    for k in k_list:
        top_k_indices = top_indices[:k]
        top_k_pairs = [(idx // n_nodes, idx % n_nodes) for idx in top_k_indices]
        
        tp = sum(1 for u, v in top_k_pairs if (u, v) in true_edges_set or (v, u) in true_edges_set)
        
        results[k] = {
            'precision': tp / k,
            'recall': tp / num_removed if num_removed > 0 else 0
        }
    return results

if __name__ == "__main__":

    results_list = []
    small_graphs = ['./fb100/data/Swarthmore42.gml', './fb100/data/Haverford76.gml', './fb100/data/Simmons81.gml', './fb100/data/Caltech36.gml', './fb100/data/Reed98.gml']
    k_list = [50, 100, 200, 400]
    fractions_f = [0.05, 0.1, 0.15, 0.2]

    for path in small_graphs:

        univ_name = path.split('/')[-1].replace('.gml', '')
        G_nx = nx.read_gml(path) 
        G_pyg = from_networkx(G_nx)

        feature_names = ['student_fac', 'gender', 'major_index', 'dorm', 'year']
        
        x_list = []
        for name in feature_names:
            if hasattr(G_pyg, name):
                val = getattr(G_pyg, name).view(-1, 1).float()
                x_list.append(val)
        
        if len(x_list) > 0:
            G_pyg.x = torch.cat(x_list, dim=-1)
        else:
            G_pyg.x = torch.eye(G_pyg.num_nodes)

        for f in fractions_f:

            transform = T.RandomLinkSplit(num_test=f, num_val=0.0, is_undirected=True)
            train_data, _, test_data = transform(G_pyg)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_features = G_pyg.num_features 

            model = GAE(GCNEncoder(G_pyg.num_features, 128, 64)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            def train():
                model.train()
                optimizer.zero_grad()
                z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
                loss = model.recon_loss(z, train_data.pos_edge_label_index.to(device))
                loss.backward()
                optimizer.step()
                return float(loss)
        
            model.train()
            for epoch in range(1000):
                optimizer.zero_grad()
                z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
                loss = model.recon_loss(z, train_data.edge_label_index.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                z_final = model.encode(test_data.x.to(device), train_data.edge_index.to(device))
                res_gnn = evaluate_gnn_top_k(z_final, train_data, test_data, k_list)
                
                for k, scores in res_gnn.items():
                    results_list.append({
                        "Graph": univ_name,
                        "f": f,
                        "Metric": "GNN (GAE)", 
                        "k": k,
                        "Precision": scores['precision'],
                        "Recall": scores['recall']
                    })

    df = pd.DataFrame(results_list)
    
    sns.set_theme(style="whitegrid")

    # 1. Histogramme / Barplot comparatif : Performance moyenne par métrique
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Metric", y="Precision", palette="viridis", capsize=.1)
    plt.title("Précision moyenne globale des prédicteurs (Tous graphes confondus)")
    plt.ylabel("Precision @k (Moyenne)")
    plt.show()

    # 2. Nuage de points (Scatter Plot) : Précision vs Rappel par Métrique
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Recall", y="Precision", hue="Metric", style="Metric", alpha=0.6)
    plt.title("Compromis Précision / Rappel par Métrique")
    plt.show()

    # 3. Évolution de la performance selon la difficulté (Fraction f)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="f", y="Precision", hue="Metric", marker="o")
    plt.title("Impact du retrait d'arêtes (f) sur la Précision")
    plt.xlabel("Fraction d'arêtes retirées (f)")
    plt.show()

    # 4. Boxplot : Distribution de la performance par université
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df[df['k']==100], x="Graph", y="Precision", hue="Metric")
    plt.xticks(rotation=45)
    plt.title("Distribution de la Precision @100 par Université")
    plt.tight_layout()
    plt.show()
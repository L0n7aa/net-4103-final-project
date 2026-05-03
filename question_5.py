import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
if __name__ == "__main__":

    small_graphs = ['./fb100/data/Swarthmore42.gml', './fb100/data/Haverford76.gml', './fb100/data/Simmons81.gml', './fb100/data/Caltech36.gml', './fb100/data/Reed98.gml']
    results = []

    for path in small_graphs:
        G_nx = nx.read_gml(path)
        attributes_to_test = ["dorm", "major_index", "gender"]
        missing_fractions = [0.1, 0.2, 0.3]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def prepare_data_for_attribute(G, target_attr, missing_fraction):
            # 1. Extraire les labels et filtrer les nœuds sans valeurs (missing data)
            labels = []
            valid_nodes = []
            for node, data in G.nodes(data=True):
                if target_attr in data and data[target_attr] != 0: # Souvent 0 = manquant dans FB100
                    labels.append(data[target_attr])
                    valid_nodes.append(node)
                    
            sub_G = G.subgraph(valid_nodes).copy()
            G_pyg = from_networkx(sub_G)
            
            # Encoder les labels en entiers de 0 à C-1
            le = LabelEncoder()
            y_encoded = le.fit_transform([sub_G.nodes[n][target_attr] for n in sub_G.nodes])
            G_pyg.y = torch.tensor(y_encoded, dtype=torch.long)
            
            # Création des features X (Matrice Identité si on se base uniquement sur la topologie)
            G_pyg.x = torch.eye(G_pyg.num_nodes, dtype=torch.float)
            
            # Création des masques
            num_nodes = G_pyg.num_nodes
            indices = np.random.permutation(num_nodes)
            test_size = int(missing_fraction * num_nodes)
            
            test_idx = indices[:test_size]
            train_idx = indices[test_size:]
            
            G_pyg.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            G_pyg.train_mask[train_idx] = True
            
            G_pyg.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            G_pyg.test_mask[test_idx] = True
            
            return G_pyg.to(device), len(le.classes_)


        for attr in attributes_to_test:
            print(f"\n--- Attribut : {attr} ---")
            for frac in missing_fractions:
                data, num_classes = prepare_data_for_attribute(G_nx, attr, frac)
                
                model = GCN(num_node_features=data.num_nodes, num_classes=num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                
                # Entraînement
                model.train()
                for epoch in range(200):
                    optimizer.zero_grad()
                    out = model(data)
                    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()
                    
                model.eval()
                with torch.no_grad():
                    pred = model(data).argmax(dim=1)
                    y_true = data.y[data.test_mask].cpu().numpy()
                    y_pred = pred[data.test_mask].cpu().numpy()
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='macro')
                    
                    results.append({
                        "University": path.split('/')[-1],
                        "Attribute": attr,
                        "Fraction": frac,
                        "MAE": mae,
                        "Accuracy": acc,
                        "Macro F1": f1  
                })

    df_final = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")

    # Graphique 1 : MAE
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_final, x="Fraction", y="MAE", hue="Attribute", marker="o")
    plt.title("MAE moyenne par attribut et fraction")
    plt.ylabel("MAE")
    plt.xlabel("Fraction de labels retirés (f)")
    plt.show()

    # Graphique 2 : Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_final, x="Fraction", y="Accuracy", hue="Attribute", marker="o")
    plt.title("Précision (Accuracy) moyenne par attribut et fraction")
    plt.ylabel("Accuracy Moyenne")
    plt.xlabel("Fraction de labels retirés (f)")
    plt.show()

    # Graphique 3 : Macro F1
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_final, x="Fraction", y="Macro F1", hue="Attribute", marker="s", linestyle="--")
    plt.title("Performance réelle (Macro F1) moyenne par attribut")
    plt.ylabel("Macro F1 Moyen")
    plt.xlabel("Fraction de labels retirés (f)")
    plt.show()
import numpy as np
import networkx as nx
from math import log
from sklearn.base import BaseEstimator, ClassifierMixin

# Question 4.b)

class LinkPrediction(BaseEstimator, ClassifierMixin):
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

class MyLinkPrediction(LinkPrediction):
    
    def __init__(self, G, metric='adamic_adar'):
        super().__init__(G)
        self.metric = metric

    def _get_common_neighbors(self, u, v):
        neighbors_u = set(self.G.neighbors(u))
        neighbors_v = set(self.G.neighbors(v))
        return neighbors_u.intersection(neighbors_v)

    def common_neighbors_score(self, u, v):
        return len(self._get_common_neighbors(u, v))

    def jaccard_score(self, u, v):
        cn = self._get_common_neighbors(u, v)
        union = set(self.G.neighbors(u)).union(set(self.G.neighbors(v)))
        if not union:
            return 0.0
        return len(cn) / len(union)

    def adamic_adar_score(self, u, v):
        cn = self._get_common_neighbors(u, v)
        score = 0.0
        for w in cn:
            degree = self.G.degree(w)
            if degree > 1:
                score += 1.0 / log(degree)
        return score

    def predict_proba(self, X):
        scores = []
        for u, v in X:
            if self.metric == 'common_neighbors':
                scores.append(self.common_neighbors_score(u, v))
            elif self.metric == 'jaccard':
                scores.append(self.jaccard_score(u, v))
            elif self.metric == 'adamic_adar':
                scores.append(self.adamic_adar_score(u, v))
            else:
                raise ValueError("Métrique inconnue")
        return np.array(scores)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > 0).astype(int)

# Question 4.c)

import random

def evaluate_link_prediction(G, fraction_removed, k_values, metric_func):
    """
    Évalue un prédicteur de liens selon le protocole de la question 4.c.
    
    :param G: Le graphe d'origine (ex: Facebook100)
    :param fraction_removed: La fraction f d'arêtes à retirer (ex: 0.1)
    :param k_values: Liste des valeurs k pour le top@k (ex: [50, 100, 200, 400])
    :param metric_func: La fonction de score à utiliser (ex: Adamic/Adar)
    :return: Un dictionnaire contenant Precision@k et Recall@k
    """
    edges = list(G.edges())
    num_to_remove = int(len(edges) * fraction_removed)
    
    # On choisit aléatoirement les arêtes à retirer
    removed_edges_list = random.sample(edges, num_to_remove)
    
    # On normalise les arêtes (u,v) de sorte que u < v pour faciliter les intersections
    removed_edges = set((u, v) if u < v else (v, u) for u, v in removed_edges_list)
    
    # graphe d'entraînement 
    G_train = G.copy()
    G_train.remove_edges_from(removed_edges_list)
    
    # ÉTAPE 3 : Calcul des scores pour les paires de nœuds
    non_edges = nx.non_edges(G_train)
    
    predictions = []
    for u, v in non_edges:
        score = metric_func(G_train, u, v)
        pair = (u, v) if u < v else (v, u)
        predictions.append((score, pair))
        
    # Tri selon le score (indice 0 du tuple)
    predictions.sort(key=lambda x: x[0], reverse=True)
    
    results = {}
    
    for k in k_values:
        # On prend les k meilleures paires prédites
        top_k_predictions = set(pair for score, pair in predictions[:k])
        
        # Intersection : Vrais Positifs (TP)
        true_positives = removed_edges.intersection(top_k_predictions)
        TP = len(true_positives)
        
        # Calcul des métriques
        # Note : Dans ce contexte, Precision@k est équivalent au top@k predictive rate
        precision_k = TP / k
        recall_k = TP / len(removed_edges)
        
        results[k] = {
            'precision': precision_k,
            'recall': recall_k,
            'TP': TP
        }
        
    return results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    results_list = []
    small_graphs = ['./fb100/data/Swarthmore42.gml', './fb100/data/Haverford76.gml', './fb100/data/Simmons81.gml', './fb100/data/Caltech36.gml', './fb100/data/Reed98.gml']
    

    # Lien entre la classe et le script ici
    
    def adamic_adar(graph, u, v):
        predictor = MyLinkPrediction(graph, metric='adamic_adar')
        return predictor.adamic_adar_score(u, v)

    def jaccard(graph, u, v):
        predictor = MyLinkPrediction(graph, metric='jaccard')
        return predictor.jaccard_score(u, v)
        
    def common_neighbors(graph, u, v):
        predictor = MyLinkPrediction(graph, metric='common_neighbors')
        return predictor.common_neighbors_score(u, v)
    
    for path in small_graphs:
        univ_name = path.split('/')[-1].replace('.gml', '')
        print(f"Analyse du graphe : {univ_name}...")
        try:
            G = nx.read_gml(path)
            n = G.number_of_nodes()
            print(f"Université : {univ_name} | Nœuds : {n} | Paires potentielles : {n**2}")
        except Exception as e:
            print(f"Erreur sur {univ_name}: {e}")
            continue

        fractions_f = [0.05, 0.1, 0.15, 0.2]
        k_list = [50, 100, 200, 400]
        
        for f in fractions_f:
            # Évaluation groupée
            metrics_to_test = [
                ("Adamic/Adar", adamic_adar),
                ("Jaccard", jaccard),
                ("Common Neighbors", common_neighbors)
            ]
            
            for m_name, m_func in metrics_to_test:
                res = evaluate_link_prediction(G, fraction_removed=f, k_values=k_list, metric_func=m_func)
                
                # Stockage pour DataFrame
                for k, scores in res.items():
                    results_list.append({
                        "Graph": path,
                        "f": f,
                        "Metric": m_name,
                        "k": k,
                        "Precision": scores['precision'],
                        "Recall": scores['recall']
                    })


    # --- PARTIE AFFICHAGE ET GRAPHIQUES ---
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
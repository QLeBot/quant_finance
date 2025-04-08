"""
En finance, le skew (ou skewness, en anglais) désigne l’asymétrie de la distribution des rendements d’un actif ou d’un portefeuille.

1. Définition mathématique du skew (ou asymétrie)
C’est une mesure statistique qui indique si la queue de la distribution des rendements est plus longue d’un côté que de l’autre :
- Skew positif : la queue est plus longue à droite. Cela signifie qu’il y a des chances plus faibles, mais possibles, de rendements très élevés.
- Skew négatif : la queue est plus longue à gauche. Cela signifie qu’il y a un risque accru de rendements très faibles (pertes importantes), même si ce sont des événements rares.
- Skew = 0 signifie que la distribution est symétrique (comme la courbe normale, en cloche).

2. Interprétation en finance
Dans la réalité, les rendements des actifs ne suivent pas une distribution normale. Le skew permet d’en tenir compte, notamment pour :
- Évaluer le risque d’un actif ou d’un portefeuille
- Améliorer les modèles de pricing d’options (ex : Black-Scholes suppose une distribution normale)
- Comprendre le comportement du marché
Par exemple :
- Les actions ont souvent un skew négatif : les pertes extrêmes sont plus fréquentes que ce que la normale prédit.
- Certains actifs ou stratégies (comme les options d’achat sur actions technologiques) peuvent avoir un skew positif, car ils ont un potentiel de gain important avec une probabilité faible.

3. Skew dans les options : le "volatility skew"
Dans le monde des options, on parle souvent de volatility skew ou vol skew, qui représente la manière dont la volatilité implicite varie selon le prix d’exercice (strike) :
- Normalement, la volatilité implicite devrait être constante pour tous les strikes.
- En réalité, elle varie, souvent à cause de la perception du risque par les investisseurs.
- Le skew des options est un outil important pour les traders d’options, car il reflète les anticipations du marché sur les extrêmes (krachs ou rallys).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, norm

# Générer trois distributions de rendements
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=10000)
data_pos_skew = np.random.lognormal(mean=0, sigma=0.7, size=10000) - 1  # pour centrer autour de 0
data_neg_skew = -np.random.lognormal(mean=0, sigma=0.7, size=10000) + 1

# Calcul des skewness
skew_normal = skew(data_normal)
skew_pos = skew(data_pos_skew)
skew_neg = skew(data_neg_skew)

# Création du graphique
plt.figure(figsize=(14, 6))

# Distribution normale
plt.subplot(1, 3, 1)
plt.hist(data_normal, bins=50, density=True, alpha=0.6, color='blue')
plt.title(f'Distribution Normale\nSkew = {skew_normal:.2f}')
plt.xlabel('Rendements')
plt.ylabel('Densité')

# Skew positif
plt.subplot(1, 3, 2)
plt.hist(data_pos_skew, bins=50, density=True, alpha=0.6, color='green')
plt.title(f'Skew Positif\nSkew = {skew_pos:.2f}')
plt.xlabel('Rendements')

# Skew négatif
plt.subplot(1, 3, 3)
plt.hist(data_neg_skew, bins=50, density=True, alpha=0.6, color='red')
plt.title(f'Skew Négatif\nSkew = {skew_neg:.2f}')
plt.xlabel('Rendements')

plt.tight_layout()
plt.show()

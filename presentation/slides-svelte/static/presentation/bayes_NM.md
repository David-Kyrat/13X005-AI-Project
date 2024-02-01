<!--
<div class="r-hstack">
<div class="r-vstack text-left">



</div>


<div class="r-vstack">



</div>
</div>
-->

# Naive Bayes

---

## Naive Bayes

- Classification probabiliste conditionnelle: théorème de Bayes : $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$
    - Features $X_0, \cdots, X_{K-1}$,  Classes $Y_0, \cdots, Y_{C-1}\qquad$ (ici $C=3, K=3$)
- Sepal length $\perp$ sepal width $\perp$ petal length $\perp$ petal width (Hypthèse d'indépendence naïve)

- Calculer la distribution empirique des features indépendemment des autres
    - $X_i$ continue $\Rightarrow$ $X_i \sim \mathcal{N}(\mu_i, \sigma_i)$, i.e. pour notre colonne de data, on calcule, la moyenne et standard deviation $\Rightarrow$ on dit que ce sont les paramètres de la loi normale qui modélise comment les données de la colonne $i$ sont réparties
    - $X_i$ binaire $\Rightarrow$ $X_i \sim \mathcal{B}(p_i)$ 
- On calcul l'impact qu'ont les V.A. de lois inférés de la répartition de chaque colonne sur le label que l'on veut prédire. 
<br>E.g. Comment la répartition de $X_2$ (longueur du pétale) nous donne une information sur le type de fleur?

- Intuitivement $\Rightarrow$ Comment la probabilité que la longueure du pétale aie une certaine valeur influe sur le type de fleur $Y_i$?

- En se prenant l'information non pas donnée par la répartition de $X_2$ mais par la répartition de tous les $[X_{j \in [0,3]}]$ $\Rightarrow$ on obtient le principe du classifier bayesien. (Chaque $X_j$ a un poids équivalent)

---

## Naive Bayes - Formellement
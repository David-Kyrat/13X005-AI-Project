---
title: Cost function
author: Michel Donnet
date: \today
header-includes:
    - \DeclareMathSymbol{*}{\mathbin}{symbols}{"01}
---


# Fonction de coût pour la régression logistique

Afin d'entraîner les paramètres de la régression logistique, il faut pouvoir comparer les résultats obtenus par la régression avec les résultats attendus.

On souhaite définir une fonction à minimiser permettant de trouver les paramètres optimaux de la régression logistique.

Notre classification se base sur la fonction sigmoïde $\sigma(z) = \frac{1}{1 + e^{-z}}$.

Comme la fonction exponnentielle est toujours positive, on a bien que $\sigma(z) \in [0, 1]$.

La fonction sigmoïde nous donne la probabilité que l'élément donné appartienne à un label.

Autrement dit, la fonction sigmoïde est la fonction de répartition de la régression logistique.

Soit $Y \in \{0, 1\}$ les différents labels que peut prendre l'élément que l'on considère et soit $X$ l'ensemble des caractéristiques connues de l'élément, dont on cherche à déterminer dans quelle classe le mettre, donc quel label on doit lui attribuer.
Soit $\theta$ le vecteur des poids des covariables, indiquant à quel point les covariables influencent sur la décision du label. On a donc:

$$P(Y = 1 | X) = \frac{1}{1 + e^{-(X_1 * w_1 + X_2*w_2 + \dots + b)}}$$
et 
$$P(Y = 0 | X) = 1 - \frac{1}{1 + e^{-(X_1 * w_1 + \dots + b)}}$$

Pour plus de simplicité, on va considérer que le biais est compris dans les poids: au lieu d'écrire $z = wX + b$, on écrit $z = \theta\hat{X}$ avec $\hat{X} = \begin{bmatrix} 1 \\ X \end{bmatrix}$ modifié ou on a ajouté un $1$ au début et $\theta = \begin{bmatrix} b & w \end{bmatrix}$.
Ainsi, on a:  
$\theta \hat{X} = \theta_0 * 1 + w X = \theta_0 + \sum_{i = 1}^n{w_i * X_i} = b + X_1 * w_1 + \dots + X_n * w_n$

Pour la suite, on va noter $X = \hat{X}$

<!-- \vspace{-3cm} -->

Notre régression logistique binaire peut donc s'écrire comme:
$$P(Y = 1 | X) = \frac{1}{1 + e^{\theta X}} = \sigma(\theta X)$$
et
$$P(Y = 0 | X) = 1 - \sigma(\theta X)$$

## Généralisation

On désire donc trouver une nouvelle distribution $\phi(z)$ tel que:
$$\phi(z) \in [0, 1]\ \forall z$$
est une généralisation de la fonction $\sigma(z)$

On veut donc que pour une régression logistique binaire, on ait $\sigma(z) = \phi(z)$.

On peut remarquer que:

$$P(Y = 1 | X)$$
$$=\frac{1}{1 + e^{-\theta X}}$$
$$=\frac{1}{1 + e^{-\theta X}} * \frac{e^{\theta X}}{e^{\theta X}}$$
$$=\frac{e^{\theta X}}{e^{\theta X} + e^{\theta X - \theta X}}$$
$$=\frac{e^{\theta X}}{e^{\theta X} + e^0}$$
$$=\frac{e^{\theta X}}{e^{\theta X} + 1}$$

On peut considérer que nous avons un vecteur de poids pour chaque label.

Ainsi, on a $\theta_0$ pour le label 0 et $\theta_1$ pour le label 1.

Comme on a besoin seulement d'un vecteur de poids pour déterminer le label de nouveaux éléments avec leurs caractéristiques, on peut considérer que $\theta_0 = 0$.

Ainsi, la formule précédente nous donne:

$$P(Y = 1 | X)$$
$$=\frac{e^{\theta_1X}}{e^{\theta_1X} + 1}$$
$$=\frac{e^{\theta_1X}}{e^{\theta_1X} + e^0}$$
$$=\frac{e^{\theta_1X}}{e^{\theta_1X} + e^{0 * X}}$$
$$=\frac{e^{\theta_1X}}{e^{\theta_1X} + e^{\theta_0X}}$$
$$=\frac{e^{\theta_1X}}{\sum_{i = 0}^1 e^{\theta_iX}}$$

On peut donc généraliser cette formule pour $K$ labels.

Cela nous donne:

$$P(Y = k| X )=\frac{e^{\theta_kX}}{\sum_{i = 0}^K e^{\theta_iX}}$$

Comme la fonction exponentielle est toujours positive, on a bien que:
$$0 \leq e^{\theta_kX} \leq e^{\theta_kX} + \sum_{i \neq k}^K e^{\theta_iX}$$
$$\Leftrightarrow 0 \leq e^{\theta_kX} \leq \sum_{i}^K e^{\theta_iX}$$
$$\Leftrightarrow 0 \leq \frac{e^{\theta_kX}}{\sum_{i}^K e^{\theta_iX}} \leq 1$$
$$\Leftrightarrow 0 \leq \phi(z) \leq 1$$

De plus, on a que:
$$\sum_k^K P(Y = k | X)$$
$$=\sum_k^K \frac{e^{\theta_kX}}{\sum_i^K e^{\theta_iX}}$$
$$=\frac{\sum_k^Ke^{\theta_kX}}{\sum_i^K e^{\theta_iX}}$$
$$=\frac{\sum_i^Ke^{\theta_iX}}{\sum_i^K e^{\theta_iX}}$$
$$=1$$

Donc la fonction $\phi(z)$ est bien une fonction de distribution de probabilité qui généralise la fonction sigmoïde pour des problèmes à plusieurs labels.

Cette fonction est courramment appelée fonction `softmax`.

Notre objectif est donc de trouver une fonction de coût pour pouvoir entraîner les paramètres de la régression multinomiale.
On cherche à maximiser la vraisemblance des données.
Donc pour un label $Y$ donné, on veut maximiser:
$$\sum_k^K f(Y, k) P(Y = k | X)$$
avec $f(Y, k)$ la fonction qui vaut $1$ si $Y = k$ et $0$ sinon.

En effet, en maximisant cette fonction, on fait en sorte que le paramètre $\theta_k$ permette d'obtenir la prédiction que le label soit égal à $k$ avec la probabilité la plus grande possible.
Afin de pouvoir utiliser un algorithme comme la descente en gradient, il faut non pas maximiser une fonction, mais minimiser une fonction.

C'est pourquoi, on peut utiliser l'inverse de cette fonction, dont on prend le logarithme pour simplifier les calculs, car on travaille avec des exponentielles.

Cela nous donne une fonction de coût comme suit:

$$\sum_k^K f(Y, k) \log(\frac{1}{P(Y = k | X)})$$
$$\sum_k^K f(Y, k) (\log(1) - \log(P(Y = k | X)))$$
$$-\sum_k^K f(Y, k)\log(P(Y = k | X))$$

On peut minimiser cette fonction de coût grâce à une descente en gradient.

Pour n données d'apprentissage, notre minimisation devient:
$$-\sum_i^n\sum_k^K f(Y_i, k) \log(P(Y_i = k | X))$$

Cette fonction de coût s'appele communément `negative log-likelihood`

## Dérivée

On va calculer la dérivée de la fonction de coût.

On a:

$$log(P(Y = k | X))$$
$$=log(\frac{e^{\theta_kX}}{\sum_i^K e^{\theta_iX}})$$
$$=\theta_kX - log(\sum_i^K e^{\theta_iX})$$

Donc:

$$\frac{\partial}{\partial \theta_{k ,j}} \sum_i^K f(Y, i)log(P(Y = i | X))$$
$$=\frac{\partial}{\partial \theta_{k ,j}} f(Y, k)log(P(Y = k | X))\ \text{  (NB: On considère que Y = k)}$$
$$=\frac{\partial}{\partial \theta_{k ,j}} f(Y, k)(\theta_{kj}X - log(\sum_i^K e^{\theta_iX})) $$
$$=f(Y, k) (x_j - \frac{\partial}{\partial \theta_{k ,j}}log(\sum_i^K e^{\theta_iX})) $$
$$=f(Y, k) (x_j - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{k ,j}}\sum_i^K e^{\theta_iX} $$
$$=f(Y, k) (x_j - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{k ,j}}e^{\theta_kX} $$
$$=f(Y, k) (x_j - \frac{x_j e^{\theta_kX}}{\sum_i^K e^{\theta_iX}}$$
$$=f(Y, k) (x_j - x_j P(Y = k | X))$$
$$=x_j (f(Y, k) - P(Y = k | X))$$

$$\frac{\partial}{\partial \theta_{j}} \sum_i^K f(Y, i)log(P(Y = i | X))$$
$$=\frac{\partial}{\partial \theta_{j}} f(Y, k)log(P(Y = k | X))\ \text{  (NB: On considère que Y = k)}$$
$$=\frac{\partial}{\partial \theta_{j}} (\theta_{k}X - log(\sum_i^K e^{\theta_iX})) $$
Supposons que j = k.
$$=X - \frac{\partial}{\partial \theta_{j}}log(\sum_i^K e^{\theta_iX}) $$
$$=X - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{j}}\sum_i^K e^{\theta_iX} $$
$$=X - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{j}}e^{\theta_jX}$$
$$=X - \frac{X e^{\theta_jX}}{\sum_i^K e^{\theta_iX}}$$
$$=X - X P(Y = j | X)$$
$$=X (1 - P(Y = j | X))$$
Supposons que $j \neq k$.

$$\frac{\partial}{\partial \theta_{j}} (\theta_{k}X - log(\sum_i^K e^{\theta_iX})) $$
$$= - \frac{\partial}{\partial \theta_{j}}log(\sum_i^K e^{\theta_iX}) $$
$$= - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{j}}\sum_i^K e^{\theta_iX} $$
$$= - \frac{1}{\sum_i^K e^{\theta_iX}} \frac{\partial}{\partial \theta_{j}}e^{\theta_jX} $$
$$= - \frac{Xe^{\theta_jX}}{\sum_i^K e^{\theta_iX}} $$
$$= -X P(Y = j | X)$$

On a donc:
$$\frac{\partial}{\partial \theta_{j}} \sum_i^K f(Y, i)log(P(Y = i | X)) = X(f(Y, j) - P(Y = j|X))$$

car $f(Y, k)$ est égal à 1 si $Y = k$ et 0 sinon.



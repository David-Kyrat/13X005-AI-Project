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

Pour plus de simplicité, on va considérer que le biais est compris dans les poids: au lieu d'écrire $z = w^TX + b$, on écrit $z = \theta^T\hat{X}$ avec $\hat{X} = \begin{bmatrix} 1 \\ X \end{bmatrix}$ modifié ou on a ajouté un $1$ au début et $\theta = \begin{bmatrix} b \\ w^T \end{bmatrix}$.
Ainsi, on a:  
$\theta^T \hat{X} = \theta_0 * 1 + w^T X = \theta_0 + \sum_{i = 1}^n{w_i * X_i} = b + X_1 * w_1 + \dots + X_n * w_n$  

<!-- \vspace{-3cm} -->

Notre régression logistique binaire peut donc s'écrire comme:
$$P(Y = 1 | X) = \frac{1}{1 + e^{\theta^T X}} = \sigma(\theta^TX)$$
et
$$P(Y = 0 | X) = 1 - \sigma(\theta^TX)$$

## Généralisation

On désire donc trouver une nouvelle distribution $\phi(z)$ tel que:
$$\phi(z) \in [0, 1]\ \forall z$$
est une généralisation de la fonction $\sigma(z)$

On veut donc que pour une régression logistique binaire, on ait $\sigma(z) = \phi(z)$.

On peut remarquer que:

$$P(Y = 1 | X)$$
$$=\frac{1}{1 + e^{-\theta^TX}}$$
$$=\frac{1}{1 + e^{-\theta^TX}} * \frac{e^{\theta^TX}}{e^{\theta^TX}}$$
$$=\frac{e^{\theta^TX}}{e^{\theta^TX} + e^{\theta^TX - \theta^TX}}$$
$$=\frac{e^{\theta^TX}}{e^{\theta^TX} + e^0}$$
$$=\frac{e^{\theta^TX}}{e^{\theta^TX} + 1}$$

On peut considérer que nous avons un vecteur de poids pour chaque label.

Ainsi, on a $\theta_0$ pour le label 0 et $\theta_1$ pour le label 1.

Comme on a besoin seulement d'un vecteur de poids pour déterminer le label de nouveaux éléments avec leurs caractéristiques, on peut considérer que $\theta_0 = 0$.

Ainsi, la formule précédente nous donne:

$$P(Y = 1 | X)$$
$$=\frac{e^{\theta_1^TX}}{e^{\theta_1^TX} + 1}$$
$$=\frac{e^{\theta_1^TX}}{e^{\theta_1^TX} + e^0}$$
$$=\frac{e^{\theta_1^TX}}{e^{\theta_1^TX} + e^{0 * X}}$$
$$=\frac{e^{\theta_1^TX}}{e^{\theta_1^TX} + e^{\theta_0^TX}}$$
$$=\frac{e^{\theta_1^TX}}{\sum_{i = 0}^1 e^{\theta_i^TX}}$$

On peut donc généraliser cette formule pour $K$ labels.

Cela nous donne:

$$P(Y = k| X )=\frac{e^{\theta_k^TX}}{\sum_{i = 0}^K e^{\theta_i^TX}}$$

Comme la fonction exponentielle est toujours positive, on a bien que:
$$0 \leq e^{\theta_k^TX} \leq e^{\theta_k^TX} + \sum_{i \neq k}^K e^{\theta_i^TX}$$
$$\Leftrightarrow 0 \leq e^{\theta_k^TX} \leq \sum_{i}^K e^{\theta_i^TX}$$
$$\Leftrightarrow 0 \leq \frac{e^{\theta_k^TX}}{\sum_{i}^K e^{\theta_i^TX}} \leq 1$$
$$\Leftrightarrow 0 \leq \phi(z) \leq 1$$

De plus, on a que:
$$\sum_k^K P(Y = k | X)$$
$$=\sum_k^K \frac{e^{\theta_k^TX}}{\sum_i^K e^{\theta_i^TX}}$$
$$=\frac{\sum_k^Ke^{\theta_k^TX}}{\sum_i^K e^{\theta_i^TX}}$$
$$=\frac{\sum_i^Ke^{\theta_i^TX}}{\sum_i^K e^{\theta_i^TX}}$$
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
$$\sum_k^K f(Y, k) - \log(P(Y = k | X))$$
$$-\sum_k^K f(Y, k)\log(P(Y = k | X))$$

On peut minimiser cette fonction de coût grâce à une descente en gradient.

Pour n données d'apprentissage, notre minimisation devient:
$$-\sum_i^n\sum_k^K f(Y_i, k) \log(P(Y_i = k | X))$$

### 2.2.1 -- Fonction de coût pour la régression logistique binaire

#### 2.2.1.1 -- Fonction de coût
Reprennons le modèle décrit dans la section 1.1.
Nous avons donc:
$$yi \sim Bernoulli(p_i), p_i = \sigma(w^T x_i + b), \sigma(z) = \frac{1}{1+e^{−z}}$$

Notre but est de calculer $p(y_i | x_i)$, puis de seuiller le résultat obtenu afin de prédire si l'élément possédant les caractéristiques $x_i$ appartient ou pas à la classe $y_i$.
On cherche donc à trouver les paramètres de poids $w$ et de biais $b$ optimaux permettant la meilleure prédiction possible.
On peut trouver la notation $p(y_i|x_i;w, b)$ indiquant que nous ne connaissons pas encore le vecteur poids $w$ et le biais $b$.

La densité de probabilité de cette fonction peut donc s'exprimer comme

$$p(y_i|x_i; w, b) = p_i^{y_i}(1 - p_i)^{1 - y_i}$$

Notre but est donc de maximiser cette fonction.
Cependant, nous préférons une fonction à minimiser plutôt qu'à maximiser, car la descente en gradient permet de trouver un minimum et non pas un maximum...

Une solution habituelle est donc d'inverser la fonction, transformant ainsi le problème de maximisation en problème de minimisation, et de prendre le logarithme de l'inverse de cette fonction afin d'éviter des valeurs extrêmes lors de notre minimisation.
L'application de la fonction logarithme sur l'inverse de la fonction est correcte car la fonction logarithme est strictement croissante, donc elle n'aura pas d'impact sur la convexité de la fonction.
Cette solution est communément appelée `Negative Logarithm Likelihood`.

Donc on cherchera à minimiser la fonction:
$$log\left(\frac{1}{p(y_i|x_i;w,b)}\right) = log(1) - log(p(y_i|x_i;w,b)) = -log(p(y_i|x_i; w, b)$$
Pour $n$ données, cette fonction peut s'écrire:
$$-\sum_i^n log(p(y_i|x_i;w,b))$$

#### 2.2.1.2 -- Dérivée de la fonction de coût

Comme nous voulons utiliser la descente en gradient, nous devons trouver la dérivée de la fonction à minimiser, donc de notre fonction de coût.

Tout d'abord, remarquons que nous pouvons écrire:
$$log(p(y_i|x_i; w, b))$$
$$= log(p_i^{y_i}(1 - p_i)^{1 - y_i})$$
$$= log(p_i^{y_i}) + log((1 - p_i)^{1 - y_i})$$
$$= y_i log(p_i) + (1 - y_i)log(1 - p_i)$$

Voici ce que nous donne la dérivée de la fonction sigmoide: 
$$\frac{d\sigma(z)}{dz}$$
$$= ((1 + e^{-z})^{-1})'$$
$$= -1 \times - e ^{-z} \times (1 + e^{-z})^{-2}$$
$$=\frac{e^{-z}}{(1 + e^{-z})^2}$$
$$=\frac{1}{1 + e^{-z}}\frac{e^{-z}}{1 + e^{-z}}$$
$$= \sigma (z) \frac{e^{-z}}{1 + e^{-z}}$$
$$= \sigma (z) \frac{1 + e^{-z} - 1}{1 + e^{-z}}$$
$$= \sigma (z) (\frac{1 + e^{-z}}{1 + e^{-z}} - \frac{1}{1 + e^{-z}})$$
$$= \sigma (z) (1 - \frac{1}{1 + e^{-z}})$$
$$= \sigma (z) (1 - \sigma (z))$$

Donc nous pouvons facilement calculer la dérivée par rapport au poid $w$ et par rapport au biais de notre fonction de coût.

Voici ce que nous donne la dérivée partielle par rapport au poids $\frac{\partial}{\partial w_j}$:

$$\frac{\partial}{\partial w_j}log(p(y_i|x_i;w,b))$$

$$=\frac{\partial}{\partial w_j} (y_i log(p_i) + (1 - y_i)log(1 - p_i))$$
$$=y_i \frac{\partial}{\partial w_j}log(p_i) + (1 - y_i)\frac{\partial}{\partial w_j}log(1 - p_i)$$
$$=y_i \frac{\partial}{\partial w_j}log(\sigma (z)) + (1 - y_i)\frac{\partial}{\partial w_j}log(1 - \sigma (z)),\ z = w^T x_i + b$$
$$=y_i \frac{1}{\sigma (z)}\frac{\partial}{\partial w_j}\sigma (z) + (1 - y_i)\frac{1}{1 - \sigma (z)}\frac{\partial}{\partial w_j}(1 - \sigma (z))$$

Or on a:
$$\frac{\partial}{\partial w_j} z = \frac{\partial}{\partial w_j}(w^T x_i + b) \Leftrightarrow \frac{dz}{\partial w_j} = x_{ij} \Leftrightarrow \frac{\partial}{\partial w_j} = \frac{d}{dz}x_{ij}$$

Donc:
$$y_i \frac{1}{\sigma (z)}\frac{\partial}{\partial w_j}\sigma (z) + (1 - y_i)\frac{1}{1 - \sigma (z)}\frac{\partial}{\partial w_j}(1 - \sigma (z))$$

$$=y_i \frac{1}{\sigma (z)}\frac{d}{dz}\sigma (z)x_{ij} + (1 - y_i)\frac{1}{1 - \sigma (z)}\left(-\frac{d}{dz} \sigma (z)x_{ij}\right)$$
$$=y_i \frac{1}{\sigma (z)}\sigma (z) (1 - \sigma (z)) x_{ij} + (1 - y_i)\frac{1}{1 - \sigma (z)}(- \sigma (z))(1 - \sigma(z))x_{ij}$$
$$=y_i (1 - \sigma (z)) x_{ij} - (1 - y_i)\sigma (z)x_{ij}$$
$$=y_i x_{ij} - y_i \sigma (z) x_{ij} + (y_i - 1)\sigma (z)x_{ij}$$
$$=y_i x_{ij} + (y_i - 1 - y_i)\sigma (z)x_{ij}$$
$$=(y_i - \sigma(z))x_{ij}$$

Voici ce que nous donne la dérivée partielle par rapport au biais $\frac{\partial}{\partial b}$:

$$\frac{\partial}{\partial b}log(p(yi|xi;w,b))$$

$$=\frac{\partial}{\partial b}(y_i log(p_i) + (1 - y_i)log(1 - p_i))$$
$$=y_i \frac{\partial}{\partial b}log(p_i) + (1 - y_i)\frac{\partial}{\partial b}log(1 - p_i)$$
$$=y_i \frac{1}{\sigma (z)}\frac{\partial}{\partial b} \sigma(z) + (1 - y_i) \frac{1}{1 - \sigma (z)} \frac{\partial}{\partial b}(1 - \sigma (z)),\ z = w^T x_i + b$$

On a:

$$\frac{\partial}{\partial b} z = \frac{\partial}{\partial b}(w^T x_i + b) \Leftrightarrow \frac{dz}{db} = 1 \Leftrightarrow \frac{\partial}{\partial b} = \frac{d}{dz}$$

Donc:
$$y_i \frac{1}{\sigma (z)}\frac{\partial}{\partial b} \sigma(z) + (1 - y_i) \frac{1}{1 - \sigma (z)} \frac{\partial}{\partial b}(1 - \sigma (z))$$
$$=y_i \frac{1}{\sigma (z)}\frac{d}{dz} \sigma(z) + (1 - y_i) \frac{1}{1 - \sigma (z)} \frac{d}{dz}(1 - \sigma (z))$$
$$=y_i \frac{1}{\sigma (z)}\frac{d}{dz} \sigma(z) - (1 - y_i) \frac{1}{1 - \sigma (z)} \frac{d}{dz}\sigma (z)$$
$$=y_i \frac{1}{\sigma (z)}\sigma(z) (1 - \sigma (z)) - (1 - y_i) \frac{1}{1 - \sigma (z)} \sigma (z)(1 - \sigma (z))$$
$$=y_i (1 - \sigma (z)) - (1 - y_i)\sigma (z)$$
$$=y_i - y_i \sigma (z) + (y_i - 1)\sigma (z)$$
$$=y_i + (y_i - 1 - y_i)\sigma (z)$$
$$=y_i - \sigma (z)$$

Donc on a:

$$\frac{\partial}{\partial w_j}log(p(y_i|x_i;w,b)) = (y_i - \sigma(z))x_{ij}$$

et:

$$\frac{\partial}{\partial b}log(p(yi|xi;w,b)) = y_i - \sigma (z)$$

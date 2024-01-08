---
title: Cost function
author: Michel Donnet
date: \today
---


# Fonction de coût pour la régression logistique

Afin d'entraîner les paramètres de la régression logistique, il faut pouvoir comparer les résultats obtenus par la régression avec les résultats attendus.


Pour cela, on pourrait penser utiliser quelque chose comme la `Mean Squared Error (MSE)`, qui est une moyenne du carré de la différence entre le résultat obtenu par la régression (donné par $z$) et la valeur estimée $y$.

La MSE nous donne une estimation de l'erreur moyenne faite entre la fonction approximative $f$ et la valeur attendue $y$.

L'objectif est donc de minimiser la MSE afin de minimiser l'erreur entre les valeurs estimées et les valeurs attendues.

Ce qui nous donnerait
$$MSE = \frac{1}{n}\sum_i^n (\sigma(z_i) - y_i)^2$$

avec $\sigma$ la fonction sigmoïde utilisée pour la régression logistique, définie comme suit:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$.

Donc notre MSE nous donnerait:

$$MSE = \frac{1}{n}\sum_i^n (\frac{1}{1 + e^{-z_i}} - y_i)^2$$

Cependant, nous pouvons remarquer que la fonction $\sigma(z)$ n'est pas linéaire.

En effet, on a $\sigma(z) = \frac{1}{1 + e^{-z}}$, ce qui n'est pas une fonction linéaire.

Cela a pour conséquence que la MSE n'est pas convexe.

La descente en gradient ne pourra donc pas fonctionnner correctement, car on pourra trouver des minimum locaux à la place du minimum global, et si on trouve un minimum local, on ne va pas trouver les paramètres optimaux pour la régression logistique.

C'est pourquoi, on utilise plutôt la log loss fonction.


Rapport de vraissemblance.....

---
title: Phénomène de sur-apprentissage
author: Michel Jean Joseph Donnet
---


# Sur-apprentissage

On ne veut pas apprendre le bruit des données d'apprentissage !

# Sur-apprentissage: exemple

![larevueia.fr](../res/overfitting_1.png){width=70%}

# Sur-apprentissage: graphique

![larevueia.fr](../res/overfitting_2.png){width=70%}

# Comment éviter le sur-apprentissage ?

Validation croisée !

![datascientest.com](../res/crossvalidation.png)

# Autres techniques ?

- Ajout données d'apprentissage modifiées (pour plus de généralisation...)
- Retirer des caractéristiques
- ...

# Concrêtement, dans le projet

Dans le projet, pour montrer le phénomène de sur-apprentissage:

- Ajout de bruits aux données d'apprentissage
- Volume réduit de données
- Modification du nombre d'itérations

# Résultats obtenus

\center ![](../res/overfitting.png){width=80%}

# Résultats obtenus

\center ![](../res/overfitting_reg.png){width=80%}

# Résultats obtenus

\center ![](../res/overfitting_reg_2.png){width=80%}

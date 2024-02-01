# Evolution des métriques

---

## Contexte :

Les mesures d'évaluations permettent d'analyser les performances de prédictions d'un modèle, à l'aide d'un "test set".

**Rappel :**

Test set : Données dont on connaît les labels exacts, que l'on cachera afin de tester les prédictions faites par le modèle.

---

## Définitions utiles :

Dans le contexte multinomial considérons un label positif et des labels nétagifs (ie. ceux qui diffèrent du label positif), on a alors :

- **True positive (TP)** : Labels positifs qui ont été correctement prédits comme tel
- **False Positive (FP)** : Labels négatifs prédits comme positifs
- **True negative (TN)** : Labels négatifs prédits comme négatifs
- **False Negative (FN)** : Labels positfs prédits comme négatifs

---

## Précision

- **Intuition** : Proportion des prédictions positives correctes (TP) par rapport à toutes les prédictions positives (TP + FP).

- **Cas multinomial** : Moyenne des précisions pour chaque label positif possible.

- **Définition** : $$\frac{1}{|L|}\cdot \sum_{l\in L}\frac{TP_l}{TP_l+FP_l}$$ où $L$ est l'ensemble des labels


---

## Rappel :

- **Intuition** : Proportion des prédictions positives correctes (TP) par rapport aux positifs réels (du test set) (TP + FN).
- **Cas multinomial** : Moyenne des rappels pour chaque label positif possible.
- **Définition Formelle** : $$\frac{1}{|L|}\cdot \sum_{l\in L}\frac{TP_l}{TP_l+FN_l}$$ où $L$ est l'ensemble des labels



---

## F1 score :

- **Intuition** : Combinaison de la précision et du rappel (moyenne harmonique)  

- **Définition** : $$\frac{2}{rappel^{-1} + precision^{-1}} = 2\cdot \frac{precision \cdot rappel }{precision + rappel}$$




---

## Accuracy :

- **Intuition** : Proportion des pŕédictions correctes parmi l’ensemble total des prédictions.

- **Définition** : $$\frac{\text{Nombre de predictions correctes}}{\text{Nombre total de prediction}}=\frac{TP + TN}{TP + TN + FP + FN}$$

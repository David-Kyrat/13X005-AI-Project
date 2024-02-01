# Analyse des features

---

## Analyse des features

<div class="r-hstack">
<div class="r-vstack text-left">

- Pic bleu et rouge faible <br /> $\Rightarrow$ $X_0$ et $X_1$ - ont moins d'influence sur la classe.

- Chevauchement faible $\Rightarrow$ peu interdépendance

</div>

<div class="r-vstack">
<img src="../src/res/comp_normal_law_Y_0.png" width="1200px">

</div>
</div>

---

## Analyse des features

<div class="r-hstack">
<div class="r-vstack text-left">

- pic bleu et vert faible <br> $\Rightarrow$ $X_0$ et $X_2$  - ont moins d'influence sur la classe.   
- Chevauchement fort entre bleu et vert et vert et rouge <br> $\Rightarrow$ interdépendance entre $X_1$ et $X_2$ et $X_0$ et $X_2$  

</div>

<div class="r-vstack">

<img src="../src/res/comp_normal_law_Y_1.png"  style="width: 1800px;">
</div>
</div>
---

## Analyse des features

<div class="r-hstack">
<div class="r-vstack text-left">

- Pic bleu et vert faible <br> $\Rightarrow$ $X_0$ et $X_2$ - ont moins d'influence sur la classe.   
- Chevauchement fort entre bleu et vert et rouge et magenta <br> $\Rightarrow$ interdépendance entre $X_1$ et $X_3$ et entre $X_0$ et $X_2$  

</div>
<div class="r-vstack">

<img src="../src/res/comp_normal_law_Y_2.png"  style="width: 1800px; ">

</div>
</div>
---

### Fonctions utilisées

`plot_util.py` : modification de la fonction plot_vs afin de pouvoir comparer jusqu'à 4 fonctions. 

`feature_analyse_plot.py` : affichage pour chaque classe  les courbes des normal PDF de chaque données.



---
title: Analyse des données
author: Noah Aviel Peterschmitt
---


<div style="display: flex; align-items: center;">
    <img src="../src/res/comp_normal_law_Y_0.png" alt="Texte alternatif" style="width: 400px; height:600px; float: left;">
    <div style="margin-left: 10px;">
        <p><center>pic bleu et rouge faible => $X_0$ et $X_1$ - ont moins d'influence sur la classe. </center>  </p>
        <p> <center>Chevauchement faible => peu interdépendance </center> </p>
    </div>
    </div>
---

<div style="display: flex; align-items: center;">
    <img src="../src/res/comp_normal_law_Y_1.png" alt="Texte alternatif" style="width: 400px; height:600px; float: left;">
    <div style="margin-left: 10px;">
        <p><center>pic bleu et vert faible => $X_0$ et $X_2$ - ont moins d'influence sur la classe. </center>  </p>
        <p> <center>Chevauchement fort entre bleu et vert et vert et rouge => interdépendance entre $X_1$ et $X_2$ et $X_0$ et $X_2$ </center> </p>
    </div>
    </div>
---

<div style="display: flex; align-items: center;">
    <img src="../src/res/comp_normal_law_Y_2.png" alt="Texte alternatif" style="width: 400px; height:600px; float: left;">
    <div style="margin-left: 10px;">
        <p><center>pic bleu et vert faible => $X_0$ et $X_2$ - ont moins d'influence sur la classe. </center>  </p>
        <p> <center>Chevauchement fort entre bleu et vert et rouge et magenta => interdépendance entre $X_1$ et $X_3$ et entre $X_0$ et $X_2$ </center> </p>
    </div>
    </div>
---

plot_util.py : modification de la fonction plot_vs afin de pouvoir comparer jusqu'à 4 fonctions. 

feature_analyse_plot.py : affichage pour chaque classe  les courbes des normal PDF de chaque données.



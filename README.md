=== ENGLISH VERSION BELOW ===

### RKmodif ###

Code Python pour l'amélioration de méthode numériques existantes pour les EDO's autonomes via l'utilisation du Machine Learning et de l'analyse rétrograde (théorie du champ modifié). Approximation du champ modifié via le Machine Learning.

Etapes:

- Création de données
- Entraînement de réseaux de neurones
- Intégration numérique.

Après choix du système dynamique souhaité, fonctions permettant:

- Création de données
- Entraînement via minimisation de la Loss
- Intégration numérique
- Courbe de convergence: méthode numérique vs méthode numérique améliorée via ML
- Courbe de convergence: approximation du champ modifié par rapport au pas de temps
- Graphe d'erreur: erreur en espace entre le champ modifié appris et le champ modifié exact
- Temps de calcul vs erreur numérique: méthode numérique, méthode numérique améliorée via ML, DOPRI5 (méthode de Runge-Kutta d'ordre 5).

### RKmodif2 - Comparison_RK_modif_1_2 ###

Etude du cas particulier de la méthode d'Euler explicite avec apprentissage séparé de chaque terme du champ modifié. Comparaison avec le programme précédent.


=== ENGLISH VERSION HERE ===

### RKmodif ###

Python code for existing numerical methods improvement to autonomous ODE's, via Machine Learning and Backward Error Analysis (modified field theory). Approximation of modified field via Machine Learning.

Steps;

- Data creation
- Training of Neural Networks
- Numerical integration

After dynamical system selection, functions of the code enable to:

- Data creation
- Training via Loss minimization
- Numerical integration
- Convergence curve: numerical method vs improved numerical method via ML
- Convergence curve: approximation of modified field w.r.t. step size
- Error graph: space error between approximated modified field and exact modified field
- Computational time vs numerical error: numérical error, improved numerical error via ML, DOPRI5 (5th order Runge-Kutta method)

### RKmodif2 - Comparison_RK_modif_1_2 ###

Particular case of Forward Euler method with separate learning of each term of the modified field. Comparison with previous code.

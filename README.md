# Projet-Image

Projet Image en python sur Weighted Voronoi Stippling

Basé sur l'article suivant:
https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf

## Prérequis

Python ≥ 3.7  
Bibliothèques nécessaires :  pip install numpy matplotlib opencv-python numba tqdm

# Generating Centroidal Voronoi Diagrams

L'objectif de cette section est de générer un diagramme de Voronoï centroïdal à partir d'un nuage de points quelconque.
On obtient ainsi un diagramme de Voronoï dans lequel le centre de chaque cellule coïncide avec le centroïde des points qu'elle contient.

Pour le calcul des aire des régions de voronoi qui sont des polygones, nous avons utilisé ce papier de Paul Bourke:
https://paulbourke.net/geometry/polygonmesh/


Pour visualiser cela il faut lancer le script generate_centroidal_voronoi_diagramm.py
A la fin du script, on pourra modifier les paramètres suivants:

nb_points = 100  <-- nombre de points du nuage de points
taille_domaine = 100 <--- les points seront génerés aléatoirement avec x et y entre 0 et taille_domaine
max_iter = 100 <-- le nombre d'itérations maximum avant arrêt du script
seuil_convergence = 1e-5  <-- seuil de déplacement moyen des centroides à partir du quel on estime qu'il y a convergence

A la fin du script, on obtient:

- Une comparaison entre le diagramme de voronoi intial (non centré) et le diagramme de voronoi final (centré) sauvegardé au chemin suivant: 'resultats/centroidal_voronoi/centroidal_voronoi_comparaison.png'

- Une animation gif montrant le "centrage" itération par itération du diagramme de voronoi initial sauvegardé au chemin suivant: 'animations/animation_centroidal_voronoi_diagramme.gif'

# Stippling with Weighted Centroidal Voronoi Diagrams

Cette section implémente un algorithme de pointillisme automatique basé sur les diagrammes de Voronoï centroïdaux pondérés. À partir d’une image (en niveaux de gris ou couleur), il génère une répartition de points (stipples) qui respecte les variations de densité tonale de l’image.

Ce fichier s'appuie sur les fonctions utilitaires du fichier weighthed_voronoi_functions_stiplling.py

Exemple Utilisation:
python stippling_script.py image.jpg [options]

Argument principal:

- `image.jpg` : Chemin vers l’image à stippler (JPG ou PNG). En général 'images/nom_image.jpg'

Options courantes:

| Option                             | Description                                                               |
|------------------------------------|---------------------------------------------------------------------------|
| `--nb_points`                      | Nombre de points à générer (défaut : 20000)                               |
| `--max_iter`                       | Nombre maximal d’itérations (défaut : 150)                                |
| `--seuil_convergence`              | Seuil d'arrêt basé sur le déplacement moyen des points (défaut : 1e-4)    |
| `--taille_points w h`              | Taille minimale et maximale des points (défaut : 1.0 1.0)                 |
| `--seuil_niveau_gris`              | Seuil en dessous duquel les points ne sont pas affichés (défaut : 70)     |
| `--affichage_intermediaire`        | Affiche une animation des itérations (désactivé par défaut)               |
| `--affichage_final`                | Affiche le résultat final (activé par défaut)                             |
| `--figsize`                        | Taille de la figure de sortie (défaut : 4)                                |
| `--min_distance`                   | Distance minimale entre deux points (défaut : 1e-3)                       |


Options avancées:

| Option                                 | Description                                                           |
|----------------------------------------|-----------------------------------------------------------------------|
| `--puissance_contraste`                | Accentuation du contraste de la densité (défaut :1)                   |
| `--influence_densite_initialisation`   | Influence de la densité pour le placement initial (défaut : 1)        |
| `--densite_centroide`                  | Influence de la densité dans le calcul des centroïdes (défaut : 1)    |

Exemple Utilisation avancée:

python stippling_weighted_centroidal_voronoi_diagramm.py photo.jpg --nb_points 10000 --affichage_intermediaire --densite_centroide 3

Tous ces paramètres peuvent être aussi rensignée dans la liste de paramètres par défaut default en début du main (ligne 92)

Résultats:

Les fichiers de sortie sont enregistrés dans le dossier `resultats/stippling` sous deux formats :
- `image-stipple_densite.pdf`
- `image-stipple_densite.png`


Exemples de lignes de commande pour obtenir les résultats de notre présentation:

- python stippling_weighted_centroidal_voronoi_diagramm.py images/figure.png --nb_points 1000 --max_iter 150 --seuil_convergence 1e-4 --seuil_niveau_gris 70 --taille_points (12.0, 12.0) --affichage_intermediaire --densite_centroide 2 --figsize 5.12 --min_distance 2e-3

- python stippling_weighted_centroidal_voronoi_diagramm.py images/shoe_1300x1300_org.png --nb_points 1000 --max_iter 150 --seuil_convergence 1e-4 --seuil_niveau_gris 70 --taille_points (10.0, 10.0) --affichage_intermediaire --densite_centroide 1.5 --figsize 13.0 --min_distance 5e-3

- python stippling_weighted_centroidal_voronoi_diagramm.py images/shoe_1300x1300_org.png --nb_points 5000 --max_iter 150 --seuil_convergence 1e-4 --seuil_niveau_gris 70 --taille_points (7.0, 7.0) --affichage_final --densite_centroide 1.3 --figsize 13.0 --min_distance 3e-3

- python stippling_weighted_centroidal_voronoi_diagramm.py images/plant4h.png --nb_points 20000 --max_iter 150 --seuil_convergence 1e-4 --seuil_niveau_gris 90 --taille_points (6.0, 6.0) --affichage_final --densite_centroide 1.5 --figsize 9.6 --min_distance 1.5e-3

# Fast Stipplings

Cette section décrit une méthode rapide de pointillisme utilisant des heuristiques et des approximations pour générer des diagrammes de Voronoï centroïdaux sans itérations coûteuses, ce qui permet de produire des résultats visuellement satisfaisants avec un temps de calcul réduit.

Deux variantes principales sont proposées :

1. **Grille en fonction du niveau de gris calculé à partir de la relaxation de Lloyd** :
   - Permet une distribution régulière des points.
   - Convient bien aux zones uniformes.
   - Correspond à la méthode proposée dans le sujet.
   - Mauvaise gestion des bords.

2. **Distribution uniforme aléatoire** :
   - Plus rapide à générer.
   - Peut produire des résultats plus organiques, mais moins réguliers.

### Visualisation des niveaux

Un échantillon de 9 niveaux pré-calculés de gris peut être affiché avec le script grid_levels.py pour validation visuelle.  
Les résultats sont sauvegardés automatiquement dans le dossier :

resultats/fast_stippling/grid_levels.png

### Exemple d'utilisation selon la méthode utilisée (voronoi vs. uniform random).

python fast_stippling_voronoi.py (voronoi)
ou
python fast_stippling_upgrade.py (uniform random)

Le résultat sera sauvegardé en fonction de l'image source sous le nom :

resultats/fast_stippling/fast_stippling_voronoi_<nom>.png
ou
resultats/fast_stippling/fast_stippling_upgrade_<nom>.png

# Evaluation of the stippling with metrics

Cette partie a pour objectif d'évaluer la qualité, de manière quantitative, les résultats obtenus sur les stippling d'images en noir et blanc. On considère les métriques d'évaluation suivantes:

## Métriques Globales

- **MSE – Erreur quadratique moyenne** afin de mesurer la différence globale d'intensité entre l'image originale et le stippling.
  - Faible MSE = bonne reproduction globale des tons (densité de points bien ajustée).
  - MSE élevée = écart significatif dans la densité ou distribution des points.

- **SSIM – Indice de similarité structurelle** afin d'évaluer la ressemblance perceptive entre l'image originale et l'image en stippling en prenant en compte la luminance, le contraste et la structure locale.
  - SSIM proche de 1 = le stippling recrée fidèlement la perception de l'image
  - SSIM faible = perte de détails structurels importants

- **Distance de Chamfer** afin d'évaluer la proximité géométrique entre deux ensembles de points 
  - Distance de Chamfer faible = bonne correspondance entre les formes.
  - Distance de Chamfer grande = les ensembles sont dissemblables.

- **Corrélation de texture** afin de comparer les textures entre l'image de référence et celle en stippling en analysant la matrice de cooccurrence pour extraire des propriétés de texture comme le contraste, puis calcule la corrélation entre les deux images.
  - Corrélation proche de 1 = le stippling a reproduit les motifs texturaux de l'image
  - Corrélation faible ou négative = perte ou inversion de textures

- **Comparaison d'histogrammes** pour évaluer si la distribution globale des points de stippling correspond bien à la distribution des intensités dans l'image originale.
  - Différence d'histogrammes faible = la densité des points correspond bien à l'intensité des pixels dans l'image originale
  - Différence élevée = mauvaise correspondance entre la distribution des points et les valeurs de l'image

## Métriques Locales

- **MSE et MAE locales** calculées sur des fenêtres locales pour identifier précisément les zones où le stippling ne correspond pas bien à l'image originale.
  - Permet de visualiser une carte d'erreur locale pour détecter les régions sur/sous-représentées.

- **Carte SSIM locale** qui montre la similarité structurelle locale entre l'image originale et la carte de densité des points.
  - Aide à identifier précisément les zones où la structure n'est pas bien préservée.

- **Erreur de Lloyd** (centrage des cellules de Voronoï) afin d'évaluer la régularité et la qualité géométrique du diagramme de Voronoï sous-jacent au stippling. On calcule l'erreur entre chaque générateur de cellule (point de stippling) et le centroïde de sa cellule de Voronoï.
  - Erreur faible = bon centrage, stippling homogène.
  - Erreur élevée = distribution sous-optimale, points mal positionnés.

## Visualisations

Le script génère plusieurs visualisations pour faciliter l'interprétation des résultats :

1. **Vue d'ensemble** : image originale, image stippling rasterisée, carte de densité des points, et points de stippling superposés sur l'image originale.
2. **Comparaison d'histogrammes** : histogrammes de l'image originale inversée et de la carte de densité, ainsi que leur différence.
3. **Analyse locale** : moyennes locales de l'image originale et de la carte de densité, ainsi que la carte d'erreur locale.
4. **Carte SSIM locale** : visualisation des zones de similarité structurelle.
5. **Heatmap d'erreur globale** : différence absolue entre l'image originale inversée et la carte de densité.

Pour utiliser ces métriques, lancez le script `evaluation_metriques.py`.

À la fin du script, renseignez `ref_img_path` et `stippling_img_path` qui sont respectivement les chemins de l'image de référence en noir et blanc et de l'image en stippling. Les images à tester se trouvent dans le répertoire `images` et les résultats des stippling se trouvent dans le répertoire `résultats/stippling`.

L'exécution affichera les différentes métriques dans le terminal et générera toutes les visualisations mentionnées ci-dessus.



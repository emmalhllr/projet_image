import tqdm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil
from generate_centroidal_voronoi_diagramm import lloyd_relaxation

##############################################################
##############     CHOIX DES PARAMETRES   ####################
##############################################################

patch_size = 128         # Taille de l'image de sortie (en pixels)
num_levels = 9           # Nombre de niveaux de stipples
max_stipples = 1000      # Nombre maximal de stipples pour le niveau 0 (noir)

# Paramètres pour l'algorithme de relaxation de Lloyd
max_iter = 100
seuil_convergence = 1e-4


# Génère un nuage de points répartis suivant des hexagones
def generate_grid(N, patch_size):

    # Création des points de manière aléatoire
    initial_points = np.random.rand(N, 2)*patch_size
    
    # Définir les limites du domaine
    boundary = [0, patch_size, 0, patch_size]
    
    # Appliquer l'algorithme de relaxation de Lloyd
    relaxed_points, _ = lloyd_relaxation(initial_points, max_iter,seuil_convergence,boundary)
    
    return relaxed_points

# Fonction pour visualiser les niveaux de gris
def visualiser_niveaux():

    # Barre de progression
    bar = tqdm.tqdm(total=num_levels)
    stipple_levels = []
    for i in range(num_levels):
        N = int(max_stipples * (1 - i / (num_levels - 1)))
        points = generate_grid(N, patch_size)
        stipple_levels.append(points)
        bar.update()
    bar.close()


    # Visualisation des niveaux
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axs.flat):
        level_idx = i * (num_levels // 9)
        pts = stipple_levels[level_idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=3,color='black')
        ax.set_title(f"Niveau {level_idx} ({len(pts)} pts)")
        ax.set_xticks([])
        ax.set_yticks([]) 
    plt.tight_layout()
    plt.savefig('resultats/fast_stippling/grid_levels.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualiser_niveaux()
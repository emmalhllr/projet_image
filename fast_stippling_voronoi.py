import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from grid_levels import generate_grid

##############################################################
##############     CHOIX DES PARAMETRES   ####################
##############################################################

PATCH_SIZE = 6              # Taille des patchs (en pixels) (l)
MAX_STIPPLES = 20           # Nb de points pour le niveau noir (t=0)
LEVELS = 80                 # Nb de niveaux de gris
STIPPLE_LIBRARY = {}        # Dictionnaire pour sauvegarder les patchs

# Paramètres pour l'algorithme de relaxation de Lloyd
max_iter = 100
seuil_convergence = 1e-4

# Fichier pour stocker les stipples
STIPPLE_FILE = "stipples_library.npz"

##############################################################
####################     FONCTIONS   #########################
##############################################################

# Pre calcul des niveaux
def precompute_stipple_levels():
    for i in tqdm.tqdm(range(LEVELS), desc="Précalcul des niveaux de stipples"):
        t = i / (LEVELS - 1)
        N = int(MAX_STIPPLES * (1 - t))  # Déterminer le nombre de points pour chaque niveau
        points = generate_grid(N, PATCH_SIZE)
        STIPPLE_LIBRARY[i] = points

# Sauvegarde dans un fichier npz compressé
def save_stipple_library(filepath=STIPPLE_FILE):
    np.savez_compressed(filepath, **{f"lvl_{k}": v for k, v in STIPPLE_LIBRARY.items()})

# Chargement depuis un fichier npz
def load_stipple_library(filepath=STIPPLE_FILE):
    data = np.load(filepath)
    return {int(k.split('_')[1]): data[k] for k in data.files}


# Stippling de l'image
def render_stipple_image(img):

    H, W = img.shape
    result_points = []

    for y in range(0, H, PATCH_SIZE-1):
        for x in range(0, W, PATCH_SIZE-1):

            y1 = max(0, y - PATCH_SIZE)
            y2 = min(img.shape[0], y + PATCH_SIZE)
            x1 = max(0, x - PATCH_SIZE)
            x2 = min(img.shape[1], x + PATCH_SIZE)
            patch = img[y1:y2, x1:x2]

            # Trouver le niveau de stipple correspondant
            t = np.mean(patch) / 255.0  
            level_index = int(t * (LEVELS - 1))  
            stipple_patch = STIPPLE_LIBRARY.get(level_index, np.empty((0, 2))) 
            
            if stipple_patch.size > 0:  
                shifted = stipple_patch + np.array([x, y])  # Décalage des points selon la position du patch
                result_points.append(shifted)
                
    return np.vstack(result_points) if result_points else np.array([])

if __name__ == "__main__":

    # Charger une image en niveaux de gris
    url_image = 'images/plant2_400x400.png'
    image = cv2.imread(url_image, cv2.IMREAD_GRAYSCALE)
    image = (image).astype(np.uint8)

    # Sauvegarde
    name_image = url_image.split('/')[1].split('.')[0]
    url_sauvegarde = 'resultats/fast_stippling/fast_stippling_voronoi_'+name_image+'.png'

    # Pré-calcul des niveaux de stipples
    precompute_stipple_levels()

    # Génération de l'image stipple
    stipple_pts = render_stipple_image(image)

    # Affichage de l'image avec les points stipples
    plt.figure(figsize=(6, 6))
    plt.scatter(stipple_pts[:, 0], stipple_pts[:, 1], s=0.5, color='black')
    plt.title("Fast stippling")
    plt.axis("off")
    plt.axis('equal')
    plt.gca().invert_yaxis() 
    plt.savefig(url_sauvegarde, dpi=300, bbox_inches='tight')
    plt.show()

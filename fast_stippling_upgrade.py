import numpy as np
import matplotlib.pyplot as plt
import cv2

##############################################################
###############     CHOIX DES PARAMETRES   ###################
##############################################################

PATCH_SIZE = 2                  # Taille des patchs (en pixels)
MAX_STIPPLES = 3                # Nb de points pour le niveau noir 
LEVELS = 80                     # Nb de niveaux de gris
STIPPLE_LIBRARY = {}            # Dictionnaire pours stocker les différents points suivant le niveau

# Générer des points uniformes
def generate_stipple_points_uniform(N, patch_size):
    points = np.random.rand(N, 2) * patch_size
    return points

# Pre calcul avec des points alétoires uniformes
def precompute_stipple_levels():
    for i in range(LEVELS):
        t = i / (LEVELS - 1)
        N = int(MAX_STIPPLES * (1 - t))
        points = generate_stipple_points_uniform(N, PATCH_SIZE)
        STIPPLE_LIBRARY[i] = points

# Fast stippling de l'image
def render_stipple_image(img):
    H, W = img.shape
    result_points = []
    for y in range(0, H, PATCH_SIZE):
        for x in range(0, W, PATCH_SIZE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            t = np.mean(patch) / 255.0
            level_index = int(t * (LEVELS - 1))
            stipple_patch = STIPPLE_LIBRARY.get(level_index, np.empty((0, 2)))
            shifted = stipple_patch + np.array([x, y])
            result_points.append(shifted)
    return np.vstack(result_points)


if __name__ == "__main__":

    # Charger image
    url_image = 'images/plant2_400x400.png'
    image = cv2.imread(url_image, cv2.IMREAD_GRAYSCALE)
    image = (image).astype(np.uint8)

    #Sauvegarde
    name_image = url_image.split('/')[1].split('.')[0]
    url_sauvegarde = 'resultats/fast_stippling/fast_stippling_upgrade_'+name_image+'.png'

    # Pré-calcul
    precompute_stipple_levels()

    # Génération de l'image stipple
    stipple_pts = render_stipple_image(image)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray", alpha=0)
    plt.scatter(stipple_pts[:, 0], stipple_pts[:, 1], s=0.3, color='black')
    plt.title("Fast stipple avec génération aléatoire")
    plt.axis("off")
    plt.savefig(url_sauvegarde, dpi=300, bbox_inches='tight')
    plt.show()

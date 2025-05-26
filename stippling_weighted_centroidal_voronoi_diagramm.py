import tqdm
import weighthed_voronoi_functions_stippling
import os.path
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit, prange
import cv2


#Normalisation de la densité
@njit
def normaliser(densite):

    min, max = np.min(densite), np.max(densite)
    if max - min > 1e-5:
        densite = (densite-min)/(max-min)
    else:
        densite = np.zeros_like(densite)
    return densite


# Initialisation parallélisée basée sur la méthode probabiliste du rejet (indiquée dans l'article)
@njit
def initialisation_points(nb_points, densite_image, rejection_power=1.0):
    height, width = densite_image.shape
    points = np.empty((nb_points, 2), dtype=np.float32)
    i = 0

    # Renforcement de la densité pour l'initialisation
    densite = densite_image ** rejection_power if rejection_power != 1.0 else densite_image

    while i < nb_points:
        x = np.random.uniform(0, height)
        y = np.random.uniform(0, width)
        p = np.random.uniform(0, 1)
        u_int = int(np.floor(x))
        v_int = int(np.floor(y))

        if p < densite[u_int, v_int]:
            points[i, 0] = y
            points[i, 1] = x
            i += 1

    return points

# Evolution du mouvement (critère d'arrêt)
@njit(parallel=True)
def mouvement_moyen(old_points, new_points):
    total = 0.0
    for i in prange(old_points.shape[0]):
        dx = new_points[i, 0] - old_points[i, 0]
        dy = new_points[i, 1] - old_points[i, 1]
        total += (dx*dx + dy*dy)**0.5
    return total / old_points.shape[0]

#Main
if __name__ == '__main__':

    # Paramètres par défaut
    default = {
        "nb_points": 20000,
        "max_iter": 150,
        "seuil_convergence": 1e-4,
        "seuil_niveau_gris": 90,
        "taille_points": (6.0, 6.0),
        "affichage_intermediaire": False,
        "affichage_final": True,
        "puissance_contraste": 1,
        "influence_densite_initialisation": 1,
        "densite_centroide": 1,
        "figsize": 9.6,
        "min_distance":1.5e-3,
    }

    # Gestion des arguments de la ligne de commande
    description = "Weighted Voronoi Stippling"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("image", type=str, help="Image à traiter")
    parser.add_argument("--nb_points", type=int, default=default["nb_points"], help="Nombre de points")
    parser.add_argument("--max_iter", type=int, default=default["max_iter"], help="Nombre maxiamal d'itération")
    parser.add_argument("--seuil_convergence", type=float, default=default["seuil_convergence"], help="Seuil de convergence")
    parser.add_argument("--taille_points", type=float, nargs=2, default=default["taille_points"], help="Taille des points pour l'affichage final")
    parser.add_argument("--seuil_niveau_gris", type=int, default=default["seuil_niveau_gris"], help="Seuil pour le niveau de gris")
    parser.add_argument("--affichage_intermediaire", default=default["affichage_intermediaire"], action='store_true', help="Afficher pas à pas le calcul")
    parser.add_argument("--affichage_final", default=default["affichage_final"], action='store_true', help="Afficher le résultat")
    parser.add_argument('--figsize', metavar='w,h', type=int, default=default["figsize"], help='Taille de la figure')
    parser.add_argument("--min_distance", type=float, default=default["min_distance"], help="Distance minimale entre 2 points")

    # Renforcement de la prise en compte de la densité
    parser.add_argument('--puissance_contraste', metavar='n', type=float, default=default["puissance_contraste"],help="Facteur pour augmenter le contraste")
    parser.add_argument('--influence_densite_initialisation', metavar='n', type=float, default=default["influence_densite_initialisation"], help="Facteur pour augementer l'influence de la densite dans l'initialisation")
    parser.add_argument('--densite_centroide', metavar='n', type=float,default=default["densite_centroide"],help=" Facteur pour l'influence de la densité dans le calcul des centroïdes.")
    args = parser.parse_args()

    nom_image = args.image
    image_brg = cv2.imread(nom_image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_brg, cv2.COLOR_BGR2RGB)

    image_rgb = image_rgb/255

    # Convertir en une carte de densité
    densite_image = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]

    # On inverse et normalise la carte de densité
    densite_image = 1.0 - normaliser(densite_image)
    # On la met dans le bon sens
    densite_image = densite_image[::-1, :]

    # Augmenter l'influence de la densité de l'image si demandé
    if args.puissance_contraste!=1:
        densite_image = densite_image**args.puissance_contraste

    # Gestion de l'enregistrement des résultats
    output_dir = os.path.join("resultats/stippling")
    basename = (os.path.basename(nom_image).split('.'))[0]
    pdf_filename = os.path.join(output_dir, basename + "-stipple_densite.pdf")
    png_filename = os.path.join(output_dir, basename + "-stipple_densite.png")

    # Initialisation
    points = initialisation_points(args.nb_points, densite_image, rejection_power=args.influence_densite_initialisation)

    # Gestion des dimensions de la fenêtre
    xmin, xmax = 0, densite_image.shape[1]
    ymin, ymax = 0, densite_image.shape[0]
    bbox = np.array([xmin, xmax, ymin, ymax])
    ratio = (xmax-xmin)/(ymax-ymin)

    # Affichage interactif
    def update_points(points):
        _ , new_points = weighthed_voronoi_functions_stippling.compute_centroids(points, densite_image, bbox, min_distance=args.min_distance)
        return new_points

    if args.affichage_intermediaire:
        
        fig = plt.figure(figsize=(args.figsize, args.figsize / ratio), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1, facecolor="k", edgecolor="None")

        # Renforcement de la densité dans le calcul des centroides
        if args.densite_centroide != 1.0:
            densite_image = densite_image ** args.densite_centroide

        # Bar de chargement
        bar = tqdm.tqdm(total=args.max_iter)

        # Lloyd relaxation
        def update(frame):
            global points
            old_points = points.copy()
            points = update_points(points)
            mouvement_moyen_points = mouvement_moyen(old_points, points)
            Pi = points.astype(int)
            X = np.clip(Pi[:, 0], 0, densite_image.shape[1] - 1)
            Y = np.clip(Pi[:, 1], 0, densite_image.shape[0] - 1)
            sizes = (args.taille_points[0] + (args.taille_points[1] - args.taille_points[0]) * densite_image[Y, X])
            scatter.set_offsets(points)
            scatter.set_sizes(sizes)
            bar.update()

            if mouvement_moyen_points < args.seuil_convergence:
                print(f"\nConvergence atteinte en {frame+1} itérations.")
                plt.savefig(pdf_filename)
                plt.savefig(png_filename)
                bar.close()
                animation.event_source.stop()

        animation = FuncAnimation(fig, update, repeat=False, frames=args.max_iter - 1)
        plt.show()


    # Si on n'a pas d'affichage interactif -> affichage_final
    else:
        if args.densite_centroide != 1.0:
            densite_image = densite_image ** args.densite_centroide

        bar = tqdm.tqdm(total=args.max_iter)
        for i in range(args.max_iter):
            old_points = points.copy()
            points = update_points(points)
            mouvement_moyen_points = mouvement_moyen(old_points, points)
            bar.update()
            if mouvement_moyen_points < args.seuil_convergence:
                print(f"\nConvergence atteinte en {i+1} itérations.")
                break
        bar.close()

        
    # Plot final et enregistrement des résultats       
    if (args.affichage_final) and not args.affichage_intermediaire:
        fig = plt.figure(figsize=(args.figsize, args.figsize/ratio), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim([xmin, xmax])
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        ax.set_yticks([])
        scatter = ax.scatter(points[:, 0], points[:, 1], s=1, facecolor="k", edgecolor="None")
        Pi = points.astype(int)
        X = np.maximum(np.minimum(Pi[:, 0], densite_image.shape[1]-1), 0)
        Y = np.maximum(np.minimum(Pi[:, 1], densite_image.shape[0]-1), 0)
        sizes = (args.taille_points[0] + (args.taille_points[1]-args.taille_points[0])*densite_image[Y, X])
        scatter.set_offsets(points)
        scatter.set_sizes(sizes)

        # Sauvegarder les figures
        plt.savefig(pdf_filename)
        plt.savefig(png_filename)

        if args.affichage_final:
            plt.show()

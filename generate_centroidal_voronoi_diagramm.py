import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import matplotlib.animation as animation

# Calcul des centroids uniforme sans tenir compte de la densité de l'image
def compute_centroide(polygone):

    # On ferme le polygone 
    if not np.array_equal(polygone[0], polygone[-1]):
        polygone = np.vstack([polygone, polygone[0]])
    
    # Initialisation des variables
    aire = 0
    cx = 0
    cy = 0
    
    for i in range(len(polygone) - 1):
        
        # Calcul de l'aire pour un polygone
        aire_temp = (polygone[i, 0] * polygone[i+1, 1] - polygone[i+1, 0] * polygone[i, 1])
        aire += aire_temp
        
        # Accumulation des coordonnées pondérées
        cx += (polygone[i, 0] + polygone[i+1, 0]) * aire_temp
        cy += (polygone[i, 1] + polygone[i+1, 1]) * aire_temp
    
    # Division par 2 pour obtenir l'aire réelle
    aire = aire / 2.0
    
    # Si l'aire est nulle ou presque, retourner le centre du rectangle englobant
    # Pour éviter erreur numérique division par quelque chose de très petit
    if abs(aire) < 1e-10:
        return np.mean(polygone, axis=0)
    
    # Calcul final du centroïde
    cx = cx / (6.0 * aire)
    cy = cy / (6.0 * aire)
    
    return np.array([cx, cy])

# Application de la relaxation de Lloyd
def lloyd_relaxation(points, max_iter=10, seuil_convergence=1e-5, boundary=None):
    
    points = np.array(points)
    history = [points.copy()] 

    iteration = 0
    non_convergence = True
    
    xmin, xmax, ymin, ymax = boundary
    
    # Ajouter des points limites loin pour délimiter le diagramme de Voronoi
    domain_size = max(xmax - xmin, ymax - ymin)

    boundary_points = np.array([
        [xmin - domain_size, ymin - domain_size],
        [xmin - domain_size, ymax + domain_size],
        [xmax + domain_size, ymin - domain_size],
        [xmax + domain_size, ymax + domain_size]
    ])
    
    while (iteration<max_iter and non_convergence):
        
        # Intégrer les points limites pour le calcul de Voronoi
        extended_points = np.vstack([points, boundary_points])

        vor = Voronoi(extended_points)
        
        new_points = np.zeros_like(points)

        for j, point in enumerate(points):
            
            # Récupérer les régions de Voronoi pour le point actuel
            region_idx = vor.point_region[j]
            region_vertices_idx = vor.regions[region_idx]
            
            # Vérifier si la région n'est pas infinie
            if -1 not in region_vertices_idx and len(region_vertices_idx) > 0:
                
                # Extraire les coordonnées des sommets de la région
                region_vertices = vor.vertices[region_vertices_idx]
                
                # Calculer le centroïde de la région 
                centroid = compute_centroide(region_vertices)
                
                # Limiter le centroïde aux frontières du domaine
                centroid[0] = max(xmin, min(xmax, centroid[0]))
                centroid[1] = max(ymin, min(ymax, centroid[1]))
                
                new_points[j] = centroid
            else:
                # Si la région n'est pas valide on conserver le point original
                new_points[j] = point
        
        # Vérifier la convergence
        mouvements = np.linalg.norm(new_points - points, axis=1)
        movement_moyen = mouvements.mean() if len(mouvements) > 0 else 0.0
        points = new_points.copy()
        history.append(points.copy()) # on actualise l'historique des itérations

        if movement_moyen < seuil_convergence:
            #print(f"Convergence atteinte après {iteration+1} itérations")
            converegence = False
        
        iteration+=1
    
    return points, history

# Fonction pour visualiser les étapes de la relaxation de Llyod
def visualize_lloyd(history, iterations=None, boundary=None,chemin_sauvegarde=''):
    
    # Créer la figure avec le bon nombre de sous-graphiques
    _, axes = plt.subplots(1, len(iterations), figsize=(4 * len(iterations), 4))
    
    # Définir les limites du graphique
    if boundary:
        xmin, xmax, ymin, ymax = boundary
    else:
        all_points = np.vstack(history)
        xmin, ymin = np.min(all_points, axis=0) - 0.1
        xmax, ymax = np.max(all_points, axis=0) + 0.1

    # Afficher les itérations sélectionnées
    for i, idx in enumerate(iterations):
        points = history[idx]
        num_points = len(points)
        vor = Voronoi(points)
        
        ax = axes[i]
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=True, point_size=5)
        ax.plot(points[:, 0], points[:, 1], 'ro', markersize=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'Itération {idx} - {num_points} points')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')


# Fonction pour créer une animation et la sauvegarder en .gif
def animate_lloyd(history, boundary=None, interval=500, save_path=None):

    if boundary:
        xmin, xmax, ymin, ymax = boundary
    else:
        all_points = np.vstack(history)
        xmin, ymin = np.min(all_points, axis=0) - 0.1
        xmax, ymax = np.max(all_points, axis=0) + 0.1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Algorithme de Lloyd - Itération: 0')
    
    # Initialisation avec la première itération
    points = history[0]
    vor = Voronoi(points)
    
    # Créer les éléments à animer
    point_scatter = ax.scatter(points[:, 0], points[:, 1], c='red', s=30)
    
    def update(frame):
        # Mettre à jour le titre
        ax.set_title(f'Algorithme de Lloyd - Itération: {frame}')
        
        # Mettre à jour les points
        points = history[frame]
        point_scatter.set_offsets(points)
        
        # Mettre à jour le diagramme de Voronoi
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'Algorithme de Lloyd - Itération: {frame}')
        
        # Recalculer Voronoi
        vor = Voronoi(points)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
        ax.scatter(points[:, 0], points[:, 1], c='red', s=30)
        
        return [point_scatter]
    
    # Création de l'animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=interval, blit=False
    )
    
    # Sauvegarde de l'animation 
    if save_path:
        if not save_path.endswith(('.gif', '.mp4')):
            save_path += '.gif'
            
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000/interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000/interval)
    
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    
    ##############################################################
    ##############     CHOIX DES PARAMETRES   ####################
    ##############################################################
    
    nb_points = 100
    taille_domaine = 100
    max_iter = 100
    seuil_convergence = 1e-5

    # Création des points de manière aléatoire
    initial_points = np.random.rand(nb_points, 2)*taille_domaine
    
    # Définir les limites du domaine
    boundary = [0, taille_domaine, 0, taille_domaine]
    
    # Appliquer l'algorithme de relaxation de Lloyd
    relaxed_points, history = lloyd_relaxation(initial_points, max_iter=max_iter,seuil_convergence=seuil_convergence,boundary=boundary)
    
    # Visualisation seulement la première et la dernière itération
    visualize_lloyd(history, iterations=[0, len(history)-1], boundary=boundary,chemin_sauvegarde='resultats/centroidal_voronoi/centroidal_voronoi_comparaison.png')
    
    # Créer une animation et l'enregistrer sous forme de GIF
    animate_lloyd(history, boundary, interval=5, save_path='animations/animation_relaxation_uniform.gif')
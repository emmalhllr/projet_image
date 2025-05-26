import numpy as np
from scipy.spatial import Voronoi
from joblib import Parallel, delayed
from numba import njit, prange

# Fonction qui retourne la liste des points contenues dans le polygone
def points_contenus_polygone(polygone):

    # Vérifier si le polygone est vide
    if len(polygone) == 0:
        return np.array([])
        
    n = len(polygone)
    X, Y = polygone[:, 0], polygone[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))

    P = []
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = i, (i+1) % n
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y-y1) * (x2-x1) / (y2-y1) + x1)

        segments.sort()
        
        # Éviter les erreurs d'indexation si segments a une longueur impaire
        num_pairs = len(segments) // 2
        for i in range(num_pairs):
            x1 = int(np.ceil(segments[2*i]))
            x2 = int(np.floor(segments[2*i+1]))

            P.extend([[x, y] for x in range(x1, x2+1)])
    
    if not len(P):
        return polygone
    return np.array(P)

# Fonction pour calculer les centroides de manière uniforme, à l'aide de l'aire
@njit
def centroide_uniform(polygone):

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


def centroide_pondere(polygone, densite_image):

    # Vérifier si le polygone est vide
    if len(polygone) == 0:
        return np.array([0, 0])
        
    # Récupérer tous les points à l'intérieur du polygone
    points_internes = points_contenus_polygone(polygone)
    
    # Limiter les coordonnées aux dimensions de l'image
    points_internes[:, 0] = np.clip(points_internes[:, 0], 0, densite_image.shape[1]-1)
    points_internes[:, 1] = np.clip(points_internes[:, 1], 0, densite_image.shape[0]-1)
    
    # Convertir en coordonnées entières pour l'indexation
    x = np.floor(points_internes[:, 0]).astype(int)
    y = np.floor(points_internes[:, 1]).astype(int)
    
    # Récupérer les densités à ces coordonnées
    densite_points = densite_image[y, x]
    
    # Ajouter une dimension pour la multiplication par coordonnées
    densite_points = densite_points.reshape(-1, 1)
    
    # Si la somme des densités est trop faible, retourner le centre du polygone
    if np.sum(densite_points) < 1e-10:
        return np.mean(points_internes, axis=0)
    
    # Calculer le centroïde pondéré
    centroide = np.sum(points_internes * densite_points, axis=0) / np.sum(densite_points)
    
    return centroide

# Fonction pour parallélisation: Calcul du centroïde pondéré pour d'une région
def compute_centroid_for_region(region, vertices, density): 
    polygon = vertices[region]
    return centroide_pondere(polygon, density)

# Fonction qui permet de calculer l'ensemble des nouveaux centroids
def compute_centroids(points, density, boundaries, min_distance=None, parallel=True, n_jobs=-1):

    # Verifier que le "radius" est respecté
    if min_distance is not None:
        points = apply_point_repulsion(points, min_distance=min_distance)

    # Diagramme de Voronoi
    vor = voronoi(points, boundaries)
    regions = vor.filtered_regions
    vertices = vor.vertices

    # Lloyd relaxation
    if not parallel:
        centroids = [centroide_pondere(vertices[region], density) for region in regions]
    else:
        centroids = Parallel(n_jobs=n_jobs)(
            delayed(compute_centroid_for_region)(region, vertices, density)
            for region in regions
        )

    return regions, np.array(centroids)


# Renvoie un booleen pour chaque point pour savoir s'il appartient à l'image
@njit
def appartient_image(points,dim_image):
    return np.logical_and(
        np.logical_and(dim_image[0] <= points[:, 0], points[:, 0] <= dim_image[1]),
        np.logical_and(dim_image[2] <= points[:, 1], points[:, 1] <= dim_image[3])
    )

# Calcul du diagramme de Voronoi sur l'image entière
def voronoi(points, dim_image):

    # Verifier que le point appartient bien à l'image
    appartient = appartient_image(points,dim_image)

    # Gestion des frontières en mirroir (seule solution vraiment probante dans notre cas)
    points_center = points[appartient,:]
    points_left = np.copy(points_center)
    points_left[:, 0] = dim_image[0] - (points_left[:, 0] - dim_image[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = dim_image[1] + (dim_image[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = dim_image[2] - (points_down[:, 1] - dim_image[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = dim_image[3] + (dim_image[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0), axis=0)
    
    # Calcul de Voronoi
    vor = Voronoi(points)
    eps = 0.1
    regions = []

    for region in vor.regions:
        region_valide = True
        for indice_sommet in region:
            if indice_sommet == -1:
                region_valide = False
                break
            else:
                # Verifier que chaque sommet est bien dans l'image
                x = vor.vertices[indice_sommet, 0]
                y = vor.vertices[indice_sommet, 1]
                if not(dim_image[0]-eps <= x <= dim_image[1]+eps and
                       dim_image[2]-eps <= y <= dim_image[3]+eps):
                    region_valide = False
                    break
        
        if region !=[] and region_valide:
            regions.append(region)

    vor.filtered_regions = regions

    return vor

# Fonction pour respecter la force de répulsion soit le "radius" dans l'article
@njit(parallel=True)
def compute_repulsion_forces(points, min_dist, force_strength):
    n = points.shape[0]
    displacements = np.zeros_like(points)
    
    for i in prange(n):
        dx, dy = 0.0, 0.0
        for j in range(n):
            if i != j:
                diff = points[i] - points[j]
                dist = np.sqrt(diff[0]**2 + diff[1]**2)
                if dist < min_dist and dist > 1e-5:
                    repulsion = (min_dist - dist) / dist
                    dx += diff[0] * repulsion
                    dy += diff[1] * repulsion
        displacements[i, 0] = dx * force_strength
        displacements[i, 1] = dy * force_strength
    return displacements

@njit
def apply_point_repulsion(points, min_distance=5.0, iterations=50, force=0.1, tolerance=1e-3):
    points = points.copy()
    for _ in range(iterations):
        displacement = compute_repulsion_forces(points, min_distance, force)
        movement = np.linalg.norm(displacement)
        points += displacement
        if movement < tolerance:
            break
    return points



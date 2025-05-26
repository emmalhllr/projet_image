import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_ubyte
from scipy.stats import pearsonr
import mahotas
from scipy.ndimage import gaussian_filter

def compute_texture_correlation(img1, img2):
    img1 = img_as_ubyte(img1)
    img2 = img_as_ubyte(img2)
    
    # Calculer la matrice de coocurrence pour les deux images 
    glcm1 = mahotas.features.texture.haralick(img1)
    glcm2 = mahotas.features.texture.haralick(img2)
    
    # Extraire des caractéristiques de texture (contraste)
    contrast1 = glcm1[:, 1]  # indice 1 pour contraste dans Haralick
    contrast2 = glcm2[:, 1]
    
    # Calculer la corrélation de Pearson entre les contrastes
    corr, _ = pearsonr(contrast1, contrast2)
    
    return corr

def compute_point_density_map(stippling_points, shape, sigma=5):
    point_image = np.zeros(shape)
    for x, y in stippling_points:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            point_image[int(y), int(x)] = 1.0
    
    # Filtre gaussien pour estimer la densité
    density_map = gaussian_filter(point_image, sigma=sigma)
    
    # Normaliser pour avoir des valeurs entre 0 et 1
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    return density_map

def compute_histograms_comparison(ref_img, density_map, n_bins=50):
    
    # Calcul des histogrammes
    hist_ref, bins_ref = np.histogram(ref_img.flatten(), bins=n_bins, range=(0, 1), density=True)
    hist_density, bins_density = np.histogram(density_map.flatten(), bins=n_bins, range=(0, 1), density=True)
    
    # Normaliser les histogrammes
    hist_ref = hist_ref / hist_ref.sum() if hist_ref.sum() > 0 else hist_ref
    hist_density = hist_density / hist_density.sum() if hist_density.sum() > 0 else hist_density
    
    # Calculer la différence entre les histogrammes 
    histogram_diff = np.sum(np.abs(hist_ref - hist_density)) / 2  
    
    # Afficher les histogrammes
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar((bins_ref[:-1] + bins_ref[1:]) / 2, hist_ref, width=(bins_ref[1] - bins_ref[0]), alpha=0.6, label='Image originale')
    plt.bar((bins_density[:-1] + bins_density[1:]) / 2, hist_density, width=(bins_density[1] - bins_density[0]), 
            alpha=0.6, label='Image stippling')
    plt.xlabel('Intensité / Densité')
    plt.ylabel('Fréquence normalisée')
    plt.title('Comparaison des histogrammes')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar((bins_ref[:-1] + bins_ref[1:]) / 2, np.abs(hist_ref - hist_density), 
            width=(bins_ref[1] - bins_ref[0]), color='red', alpha=0.7)
    plt.xlabel('Intensité / Densité')
    plt.ylabel('Différence absolue')
    plt.title(f'Différence entre histogrammes (score: {histogram_diff:.4f})')
    
    plt.tight_layout()
    
    return histogram_diff, hist_ref, hist_density

def compute_local_comparison(ref_img, density_map, window_size=15):
    
    height, width = ref_img.shape
    half_window = window_size // 2
    
    # Matrices pour stocker les moyennes locales
    ref_local_means = np.zeros_like(ref_img)
    density_local_means = np.zeros_like(density_map)
    
    # Calculer les moyennes locales pour chaque pixel
    for y in range(height):
        for x in range(width):
            # Définir les limites de la fenêtre
            y_min = max(0, y - half_window)
            y_max = min(height, y + half_window + 1)
            x_min = max(0, x - half_window)
            x_max = min(width, x + half_window + 1)
            
            # Calculer les moyennes locales
            ref_local_means[y, x] = np.mean(ref_img[y_min:y_max, x_min:x_max])
            density_local_means[y, x] = np.mean(density_map[y_min:y_max, x_min:x_max])
    
    # Calculer la carte d'erreur locale
    local_error_map = np.abs(ref_local_means - density_local_means)
    
    # Calcul des métriques globales pour cette comparaison locale
    local_mse = np.mean(local_error_map ** 2)
    local_mae = np.mean(local_error_map)
    
    # Visualiser la carte d'erreur locale
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(ref_local_means, cmap='gray')
    plt.title("Moyennes locales (image originale)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(local_error_map, cmap='hot')
    plt.colorbar(label="Erreur locale")
    plt.title(f"Carte d'erreur locale (MAE: {local_mae:.4f}, MSE: {local_mse:.4f})")
    plt.axis('off')
    
    plt.tight_layout()
    
    return local_error_map, local_mse, local_mae

def compute_local_ssim_map(ref_img, density_map, win_size=11):

    # Calculer la carte SSIM
    try:
        ssim_value, ssim_map = ssim(
            ref_img, 
            density_map, 
            win_size=win_size, 
            data_range=1.0, 
            full=True
        )
    except ValueError:
        # Méthode alternative - utiliser une fenêtre glissante manuelle
        height, width = ref_img.shape
        half_win = win_size // 2
        ssim_map = np.zeros_like(ref_img)
        
        # Pour chaque région de l'image
        for y in range(half_win, height - half_win):
            for x in range(half_win, width - half_win):
                # Extraire la fenêtre
                win1 = ref_img[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                win2 = density_map[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                
                # Calculer le SSIM local pour cette fenêtre
                ssim_map[y, x] = ssim(win1, win2, win_size=win_size, data_range=1.0)
        
        ssim_value = np.mean(ssim_map)
    
    # Visualiser la carte SSIM
    plt.figure(figsize=(10, 8))
    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar(label="SSIM locale")
    plt.title(f"Carte SSIM locale (moyenne: {ssim_value:.4f})")
    plt.axis('off')
    
    return ssim_map, ssim_value

# Code de diagnostic
def debug_image_processing(image_path, stipple_path):
    # Charger les images
    original = io.imread(image_path, as_gray=True)
    stipple = io.imread(stipple_path, as_gray=True)
    
    # Informations de base sur les images
    print(f"Image originale: min={original.min()}, max={original.max()}, moyenne={original.mean()}")
    print(f"Image stippling: min={stipple.min()}, max={stipple.max()}, moyenne={stipple.mean()}")
    
    # Histogrammes bruts (sans normalisation)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(original.flatten(), bins=50, alpha=0.7, label='Original')
    plt.title("Histogramme image originale")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(stipple.flatten(), bins=50, alpha=0.7, label='Stipple')
    plt.title("Histogramme image stippling")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Extraction des points avec différents seuils
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 3))
    
    for i, threshold in enumerate(thresholds):
        points = np.argwhere(stipple > threshold)
        axes[i].imshow(stipple > threshold, cmap='gray')
        axes[i].set_title(f"Seuil > {threshold}\n{len(points)} points")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Générer des cartes de densité avec différents sigmas
    sigmas = [1, 3, 5, 8, 12]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 3))
    
    points = np.argwhere(stipple > 0.5)[:, [1, 0]]  # x, y
    
    for i, sigma in enumerate(sigmas):
        density = compute_point_density_map(points, original.shape, sigma=sigma)
        axes[i].imshow(density, cmap='gray')
        axes[i].set_title(f"Sigma = {sigma}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

###############################################################################################################
#################################### CHARGEMENT DES IMAGES ####################################################
###############################################################################################################

ref_img_path = 'images/shoe_1300x1300_org.png'
stippling_img_path = 'resultats/fast_stippling/fast_stippling_upgrade_shoe_1300x1300_org.png'

reference_image = io.imread(ref_img_path, as_gray=True)
stippling_image = io.imread(stippling_img_path, as_gray=True)

shape = reference_image.shape


reference_image = (reference_image - reference_image.min()) / (reference_image.max() - reference_image.min())

# Extraire les points de stippling de l'image
stippling_points = np.argwhere(stippling_image > 0.95)[:, [1, 0]]

# Générer une carte de densité à partir des points de stippling
density_map = compute_point_density_map(stippling_points, shape, sigma=4.0)

#Correlation
texture_correlation = compute_texture_correlation(reference_image, density_map)

# Comparaison des histogrammes
hist_diff, hist_ref, hist_density = compute_histograms_comparison(reference_image, density_map, n_bins=50)

# Comparaison locale
local_error_map, local_mse, local_mae = compute_local_comparison(reference_image, density_map, window_size=15)

# Calcul de la carte SSIM locale
ssim_map, local_ssim_value = compute_local_ssim_map(reference_image, density_map, win_size=11)

# Afficher visuellement la comparaison initiale
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.imshow(reference_image, cmap='gray')
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(density_map, cmap='gray')
plt.title("Carte de densité des points")
plt.axis('off')

plt.tight_layout()


# Afficher les résultats numériques
results = {
    "Métriques globales": {
        "Corrélation de texture": texture_correlation,
        "Différence d'histogrammes": hist_diff,
    },
    "Métriques locales": {
        
        "MSE locale": local_mse,
        "MAE locale": local_mae,
        "SSIM locale moyenne": local_ssim_value
    }
}

print("\n--- RÉSULTATS DE L'ÉVALUATION DU STIPPLING ---")
for category, metrics in results.items():
    print(f"\n{category}:")
    for name, value in metrics.items():
        # Gérer les valeurs NaN ou infinies
        if np.isnan(value) or np.isinf(value):
            print(f"  {name}: Valeur non calculable")
        else:
            print(f"  {name}: {value:.4f}")

plt.show()
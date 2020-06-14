import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model

#######################################
# UTILITAIRES POUR LE PREAMBULE
#######################################

def load_usps(filename):
    """ str -> np.array(int) * np.array(int)
        Charge la base de données USPS
        Retourne les ensembles X et Y
    """
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
        tmp = np.array(data)
    return tmp[: ,1:] ,tmp[: ,0]. astype(int)

def show_coefficients(model, name, save=False):
    """ Model -> void
        Affiche les coefficients (poids) du modèle linéaire avec une colormap viridis
    """
    bounds = np.arange(-0.2,0.16, 0.01)
    norm = mpl.colors.BoundaryNorm(bounds, plt.cm.viridis.N)
    plt.imshow((model.coef_.reshape(16,16)), cmap=plt.cm.viridis, norm=norm)
    plt.colorbar()
    if save:
        plt.savefig("figures/"+name+".png")
    plt.title("Coefficients du modèle '"+name+"'.\nNombre de non-zéros : "+str(np.count_nonzero(model.coef_)))
    plt.show()

#######################################
# OUTILS POUR LE TRAITEMENT D'IMAGES
#######################################

def read_img(filename):
    """ str -> np.array(float**3)
        Lit une image à partir de son nom de ficher
        Retourne l'image au format hsv sous forme d'un np.array dont les valeurs sont comprises entre -1 et 1
    """
    img = plt.imread(filename)[:,:,0:3]
    np.linalg.norm(img)
    img = rgb_to_hsv(img)
    img = -1 + 2*img
    return img

def to_png(img):
    """ np.array(float**3) -> np.array(float**3)
        Convertit une image au format .png pour son affichage
    """
    img_png = np.where(img == -100, -1, img)
    img_png = (1 + img_png) / 2
    img_png = hsv_to_rgb(img_png)
    return img_png

def show_img(img, title="", save=False):
    """ np.array(float**3), boolean -> void
        Affiche une image et l'enregistre au format .png si save=True
    """
    img_to_show =to_png(img)
    plt.imshow(img_to_show)
    if save :
        plt.savefig("figures/"+title+".png")
    plt.title(title)
    plt.show()

def get_patch(i,j,h,img):
    """ int, int, int, np.array(float**3) -> np.array(float**3)
        Renvoie le patch de centre (i,j) et de longueur h de l'image img
    """
    return img[i-h//2:i+h//2:,j-h//2:j+h//2]

def patch_to_vect(patch):
    """ np.array(float**3) -> np.array(float)
         Convertit un patch en vecteur
    """
    return patch.ravel(order='C')

def vect_to_patch(vect):
    """ np.array(float) -> np.array(float**3)
         Convertit un vecteur en patch
    """
    h = int(np.sqrt(vect.size / 3))
    return vect.reshape(h,h,3)

def noise(img, prc):
    """ np.array(float**3), float -> np.array(float**3)
         Bruite l'image en supprimant aléatoirement un pourcentage de ses pixels
    """
    img_noised = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            n = np.random.rand()
            if n < prc :
                img_noised[i,j] = [-100, -100, -100]
    return img_noised

def delete_rect(img,i,j,height,width):
    """ np.array(float**3), int, int, int, int -> np.array(float**3)
        Supprime un rectangle de l'image de dimension (height, witdh) et de coin supérieur gauche (i,j)
    """
    img_del = img.copy()
    img_del[i-height//2:i+height//2:,j-width//2:j+width//2] = -100
    return img_del

def miss_pixel(patch):
    """ np.array(float**3) -> boolean
        Retourne True si le patch contient au moins un pixel manquants, False sinon
    """
    return np.argwhere(patch == -100).shape[0] > 0

def empty_patch(patch):
    """ np.array(float**3) -> boolean
        Retourne True si le patch ne contient que des pixels manquants, False sinon
    """
    return np.argwhere(patch == -100).shape[0] == patch.size

def atoms_dictionary(img, h, step):
    """ np.array(float**3), int, int -> dict(patch)
        Retourne le dictionnaire des patchs de longueur h de l'image dont tous les pixels sont exprimés
        step : pas
    """
    if step == -1:
        step = h
    dic = {}
    height, width, dim = img.shape
    for i in range(h, height-h//2+1, step):
        for j in range(h, width-h//2+1, step):
            patch = get_patch(i, j, h, img)
            if not miss_pixel(patch):
                dic[i,j] = patch
    return dic

def noise_dictionary(img, h, step):
    """ np.array(float**3), int, int -> dict(patch)
        Retourne le dictionnaire des patchs de longueur h de l'image dont il manque des pixels
    """
    dic = {}
    height, width, dim = img.shape
    for i in range(h//2, height-h//2+1, step):
        for j in range(h//2, width-h//2+1, step):
            patch = get_patch(i, j, h, img)
            if miss_pixel(patch):
                dic[i,j] = patch
    return dic

def learn_weigth(patch, dictionary, alpha="best"):
    """ np.array(float**3), dict(patch), float -> list(float), np.array(float**3)
        Détermine les poids sur chaque atome du dictionnaire à l'aide de l'algorithme du LASSO
        Retourne les coefficients obtenus et le patch reconstitué
    """
    if alpha =='best':
        lasso_model = linear_model.LassoCV(max_iter = 10000)
    else:
        lasso_model = linear_model.Lasso(alpha=alpha, max_iter = 10000)
    Y = patch_to_vect(patch)
    train_i = np.argwhere(Y != -100).ravel()
    test_i  = np.argwhere(Y == -100).ravel()
    X = np.empty((patch.size, len(dictionary)))
    i = 0
    for patch_i in dictionary.values():
        X[:,i] = patch_to_vect(patch_i)
        i = i +1
    lasso_model.fit(X[train_i], Y[train_i])
    #if alpha == 'best':
        #print("alpha choosen :", lasso_model.alpha_)
    w = lasso_model.coef_
    #print("somme des poids :", w.sum())
    #print("number of non-zeros :", np.count_nonzero(lasso_model.coef_) )
    Y[test_i] = lasso_model.predict(X[test_i])
    Y = np.where(Y <= -1, -1, Y)
    Y = np.where(Y >= 1, 1, Y)
    #print(Y[test_i])
    return w, vect_to_patch(Y)

def replace_patch(img, patch, i, j, h):
    """ np.array(float**3), np.array(float**3), int, int, int -> void
        Remplace le patch de centre (i,j) et de longueur h dans l'image
    """
    img[i-h//2:i+h//2:,j-h//2:j+h//2] = patch

def dictionary_centered(img, x, y, m, n, h, step):
    """ np.array(float**3), int, int, int, int, int-> np.array(float**3)
        Retourne le dictionnaire des patchs centrés sur le rectangle de taille (m*n) et de coin supérieur gauche (i,j)
    """
    dic = {}
    for i in range(x, x+m+h, step):
        for j in range(y, y+n+h, step):
            patch = get_patch(i, j, h, img)
            if miss_pixel(patch):
                dic[i,j] = patch
    return dic

def initialize_peel(img, h, step):
    """ np.array(float**3), int, int, int, int, int-> np.array(float**3)
        Calcule la nouvelle couche de "pelure d'oignon" de l'image en retournant le dictionnaire des patchs centrés sur les pixels manquants

    """
    missing_pixels = np.argwhere(img == -100)
    x = missing_pixels[0,0]
    y = missing_pixels[0,1]
    m =  missing_pixels[-1,0] -  missing_pixels[0,0]
    n =  missing_pixels[-1,1] -  missing_pixels[0,1]
    #print(x, y, m, n)
    dic = dictionary_centered(img, x, y, m, n, h, step)
    return dic

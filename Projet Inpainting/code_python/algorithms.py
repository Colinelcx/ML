from .tools import *

#######################################
# ALGORITHMES D'INPAINTING
#######################################

def naive_reconstruction(img, h, step, alpha="best", title="", show=False):
    """ np.array(float**3), int, float -> np.array(float**3)
        Reconstruit naïvement une image en remplaçant les patchs dont il manque des pixels à l'aide de l'algorithme du LASSO
        L'ordre de reconstruction des patchs n'est pas déterminé, et si un patch ne contient aucun pixel exprimé il n'est pas reconstruit
        Si alpha = best, le meilleur alpha est calculé par cross-validation
    """
    missing_pixels = noise_dictionary(img, h, step)
    dic = atoms_dictionary(img,h, step)
    new_img = img.copy()
    for (i,j), patch in missing_pixels.items():
        if not empty_patch(patch):
            w, new_patch = learn_weigth(patch, dic, alpha)
            replace_patch(new_img, new_patch, i, j, h)
        if show :
            plt.show_img(new_img)
    return new_img

def onion_peel_reconstruction(img, h, step, alpha="best", show=False):
    """ np.array(float**3), int, float -> np.array(float**3)
        Reconstruit une image selon le principe "pelage d'oignon" : Partir des patchs du bord de la partie manquante puis remplir au fur et à mesure vers le centre de l’image
        Si alpha = best, le meilleur alpha est calculé par cross-validation
    """
    new_img = img.copy()
    dic = atoms_dictionary(img, h, step)
    stop = False
    while not stop :
        centered_dic = initialize_peel(new_img, h, step)
        stop = True
        for (i,j), patch in centered_dic.items():
            if not empty_patch(patch):
                w, new_patch = learn_weigth(patch, dic, alpha)
                replace_patch(new_img, new_patch, i, j, h)
            else:
                stop = False # un patch n'as pas pu être traité donc on continue
        if show:
            show_img(new_img)
        dic= atoms_dictionary(new_img, h, step)
    return new_img

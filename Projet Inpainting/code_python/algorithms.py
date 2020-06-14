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
        centered_dic = initialize_peel(new_img, h, step) # initialise la nouvelle "couche d'oignon"
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

def preserve_structures_reconstruction(img, h, step, alpha='best', show = False):
    """ np.array(float**3), int, float -> np.array(float**3)
        Reconstruit une image selon le principe "preservation structure" : Remplit les patch selon leur ordre de priorité : les patchs
        contenant des bordures ou ayant le plus de pixels exprimés sont traités en priorité
        Si alpha = best, le meilleur alpha est calculé par cross-validation
    """
    damaged_zone = np.argwhere(img == -100)
    x = damaged_zone[0,0]
    y = damaged_zone[0,1]
    m =  damaged_zone[-1,0] -  damaged_zone[0,0]
    n =  damaged_zone[-1,1] -  damaged_zone[0,1]
    edges = compute_edges(img, x, y, m, n)
    plt.imshow(Ig, cmap='gray')
    plt.title("Bordures de l'image")
    plt.show()
    new_img = img.copy()
    dic = atoms_dictionary(img, h, step)
    stop = False
    damaged_dic = dictionary_priority(img, x, y, m, n, h)
    while not stop :
        if show :
            show_img(new_img)
        damaged_dic = dictionary_priority(new_img, x, y, m, n, h)
        if not damaged_dic : # le dictionnaire est vide
            damaged_zone = np.argwhere(new_img == -100)
            if damaged_zone.size == 0: #l'image est reconstituée
                stop = True
                break;
            # sinon on recentre les patchs
            x = damaged_zone[0,0]
            y = damaged_zone[0,1]
            m =  damaged_zone[-1,0] -  damaged_zone[0,0]
            n =  damaged_zone[-1,1] -  damaged_zone[0,1]
            damaged_dic = dictionary_priority(new_img,x, y, m, n, h)
        keys = []
        for k in damaged_dic.keys():
            keys.append(k)
        prior = int(np.array(keys).max()) # choix du patch à traiter en priorité

        patch, i, j, confidence, data_term = damaged_dic[prior]
        #print("choosen : confidence=",confidence, "data_term=",data_term)
        w, new_patch = learn_weigth(patch, dic, alpha)
        replace_patch(new_img, new_patch, i, j, h)
        if show :
            show_img(new_img)
        dic= atoms_dictionary(new_img, h, step) #mise à jour du dictionnaire d'atomes
    return new_img

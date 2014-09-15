from yo import load_txt, get_track_objects, get_colocal_arrays
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA, FastICA
import argparse


cell_diameter = 100 #um 
cell_radius = cell_diameter / 2

def select_ca1(res, n=20, threshold=60):
    is_ca1 = []
    threshold **= 2
    for p in res:
        dist_array = np.sum((res - p[None, :]) ** 2, axis=1)
        dist_array.sort()
        is_ca1.append(dist_array[n-1] < threshold)
    return np.array(is_ca1)

def _res_2_coord(res_dic, pixel_size):
    pixel_size = np.array(pixel_size)
    coords = np.array(list(zip(res_dic["Location_Center_X"], res_dic["Location_Center_Y"], res_dic["ImageNumber"]))).T
    coords *= pixel_size[:, None]
    return coords

'''       
def get_colocal_arrays(reference, target_list, pixel_size=(0.332, 0.332, 1.0), tol=1.0):
    Match target_list objects onto reference. 
    return: a boolean numpy array of (N_reference_obj, n_target).
    ref = _dic_2_coord(reference, pixel_size)
    target_list = [_dic_2_coord(x, pixel_size) for x in target_list]
    tol **= 2
    res = (np.zeros((reference.shape[0], len(target_list))) != 0)
    for i,coord in enumerate(reference):
        for j,target in enumerate(target_list):
            res[i,j] = (np.min(np.sum((target - coord) ** 2, axis=1)) < tol)
    return res  
'''

def get_args():
    parser = argparse.ArgumentParser(\
            description = "Analysis file for Yosuke's CellProfiler result",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--target",
            help = "The two experimental cell groups to be analyzed, arc and h1a",
            nargs = "2")
    parser.add_argument("-r", "--reference",
            help = "The reference cell groups to be sampled from")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ref_txt = load_txt(args.reference)
    target_txt_list = [load_txt(x) for x in args.target]
    colocal =  get_colocal_arrays(ref_txt, target_txt_list)
    res, label = get_track_objects(ref_txt, colocal_info_array = colocal)
    pca = PCA()
    X = res
    S_pca = pca.fit(X).transform(X)
    ica = FastICA()
    S_ica = ica.fit(X).transform(X)
    S_ica /= S_ica.std(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cell_colors = ['blue', 'green', 'red', 'yellow', 'brown']
    cell_labels = ["DAPI", "Arc", "H1a", "Arc+H1a", 'non-CA1']
    plot_data = []

    is_ca1 = select_ca1(res)

    
    plot_data.append(res[~label[:,0] & ~label[:,1] & is_ca1])
    plot_data.append(res[label[:,0] & ~label[:,1] & is_ca1])
    plot_data.append(res[~label[:,0] & label[:,1] & is_ca1])
    plot_data.append(res[label[:,0] & label[:,1] & is_ca1])
    plot_data.append(res[~is_ca1])

    for plot, color, label in zip(plot_data, cell_colors, cell_labels):
        ax.scatter(plot[:,0], plot[:,1], plot[:,2], c=color, s=cell_radius, label=label, marker=".", edgecolors=color)

    axis_list = [pca.components_.T, ica.mixing_]
    colors = ['orange', 'red']
    source = [0,0,0]
    res_mean = res.mean(axis=0)
    source_list = []
    for i in range(3):
        source_list.append(np.array([res_mean[i]] * 3))
    
    for color, axis in zip(colors, axis_list):
        axis /= np.sqrt(np.sum(axis ** 2, axis=0))
        #axis *= res.std()
        print(axis)
        x,y,z = axis
        l = 200
        pca_axis = ax.quiver(source_list[0] + l*x, source_list[1] + l*y, source_list[2]+l*z,x,y,z, color=color, length = l, arrow_length_ratio=0.1)
    plt.show()




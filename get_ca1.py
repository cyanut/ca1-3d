from yo import load_txt, get_track_objects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA, FastICA


cell_diameter = 13 #um 



if __name__ == "__main__":
    import sys
    res = get_track_objects(load_txt(sys.argv[1]))
    pca = PCA()
    X = res
    S_pca = pca.fit(X).transform(X)
    ica = FastICA()
    S_ica = ica.fit(X).transform(X)
    S_ica /= S_ica.std(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.scatter(res[:,0], res[:,1], res[:,2])

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




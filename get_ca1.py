from yo import get_cells
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA, FastICA
import argparse
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap


class LinearColormap(LinearSegmentedColormap):

    def __init__(self, name, segmented_data, index=None, **kwargs):
        if index is None:
            # If index not given, RGB colors are evenly-spaced in colormap.
            index = np.linspace(0, 1, len(segmented_data['red']))
        for key, value in segmented_data.items():
            # Combine color index with color values.
            segmented_data[key] = zip(index, value)
        segmented_data = dict((key, [(x, y, y) for x, y in value])
                              for key, value in segmented_data.items())
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


# Red for all values, but alpha changes linearly from 0.3 to 1
color_spec = {'blue':  [1.0, 0.0],
           'green': [0.0, 0.0],
           'red':   [0.0, 1.0],
           'alpha': [0.05, 1.0]}
alpha_red = LinearColormap('alpha_red', color_spec)

class InteractiveFig(object):

    def __init__(self, fig, control_axes= [], plots = [], data = [], init_index=0, color="red", idx_text_offset = (5, 5), control_range = None, listening_events = ["button_press_event", "key_press_event"], event_handler=None, update_handler=None):
        assert len(plots) == len(data), "InteractiveFig: axes should match data"
        self.fig = fig
        self.control_axes = control_axes
        self.index = init_index
        self.plots = list(zip(plots, data))
        self.event_handler = event_handler
        self.update_handler = update_handler

        self.hlines = []
        for axis in control_axes:
            hline = axis.axvline(x=self.index, color=color, zorder=99)
            idx_text = axis.annotate(str(self.index), color=color, xy=(self.index, axis.get_ylim()[0]), xycoords="data", xytext=idx_text_offset, textcoords='offset points', ha='left')
            self.hlines.append((hline, idx_text))

        if control_range is None:
            range_max = min([len(d) for (p,d) in self.plots])
            self.control_range = (0, range_max - 1)
        else:
            self.control_range = control_range

        self.listening_events = listening_events
        self.cid = [self.fig.canvas.mpl_connect(event, self.on_input) for event in self.listening_events]

    def update_plot(self):
        for plot, data in self.plots:
            if isinstance(plot, matplotlib.image.AxesImage) and len(data.shape) >= 3:#images
                plot.set_data(data[...,self.index])
                plot.autoscale()
            elif isinstance(plot, matplotlib.lines.Line2D):#1d data
                plot.set_ydata(data[..., self.index])
                plot.axes.relim()
                plot.axes.autoscale_view()
            elif isinstance(plot, list) and len(plot) > 0 and isinstance(plot[0], matplotlib.lines.Line2D):#2d points
                plot[0].set_xdata(data[self.index][1])
                plot[0].set_ydata(data[self.index][0])
                if len(data[self.index]) > 2:
                    plot[0].set_color(data[self.index[2]])
            elif isinstance(plot, matplotlib.collections.PathCollection):
                plot.set_offsets([(x,y) for x, y in zip(data[self.index][0], data[self.index][1])])
                if len(data[self.index]) > 2:
                    plot.set_facecolor(data[self.index][2])
            else:
                print("Unhandled plot:", plot)
        if self.update_handler:
            self.update_handler(self)


        for hline, idx_text in self.hlines:
            hline.set_xdata(self.index)
            idx_text.xy = (self.index, idx_text.xy[1])
            idx_text.set_text(str(self.index))

        self.fig.canvas.draw()

    def on_input(self, event):
        if isinstance(event, KeyEvent):
            print(self.index, self.control_range)
            if event.key == "left" and self.index > self.control_range[0]:
                self.index -= 1
                self.update_plot()
            elif event.key == "right" and self.index < self.control_range[1]:
                self.index += 1
                self.update_plot()

        elif isinstance(event, MouseEvent) and event.inaxes in self.control_axes:
            self.index = int(event.xdata)
            self.update_plot()

        if self.event_handler:
            self.event_handler(self, event)


def select_ca1(res, n=20, threshold=60):
    #return np.zeros(res.shape[0]) == 0
    is_ca1 = []
    threshold **= 2
    for p in res:
        dist_array = np.sum((res - p[None, :]) ** 2, axis=1)
        dist_array.sort()
        is_ca1.append(dist_array[n-1] < threshold)
    return np.array(is_ca1)


def plot_3d(res, label, ca1_label, cell_colors, cell_labels):
    pca = PCA()
    X = res
    S_pca = pca.fit(X).transform(X)
    ica = FastICA()
    S_ica = ica.fit(X).transform(X)
    S_ica /= S_ica.std(axis=0)

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    plot_data = []

    plot_data.append(res[~label[:,0] & ~label[:,1] & ca1_label])
    plot_data.append(res[label[:,0] & ~label[:,1] & ca1_label])
    plot_data.append(res[~label[:,0] & label[:,1] & ca1_label])
    plot_data.append(res[label[:,0] & label[:,1] & ca1_label])
    plot_data.append(res[~ca1_label])

    for plot, color, label in zip(plot_data, cell_colors, cell_labels):
        ax.scatter(plot[:,0], plot[:,1], plot[:,2], c=color, s=args.cell_diameter / 2, label=label, marker=".", edgecolors=color)

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
        print("Axis:", axis)
        x,y,z = axis
        l = 200
        pca_axis = ax.quiver(source_list[0] + l*x, source_list[1] + l*y, source_list[2]+l*z,x,y,z, color=color, length = l, arrow_length_ratio=0.1)
    plt.show()


def get_bound(res):
    bounding_min = np.min(res, axis=0)
    bounding_max = np.max(res, axis=0)
    '''
    bounding_min -= r
    bounding_max += r
    bounding_min[bounding_min < 0] = 0
    '''
    bound = np.vstack((bounding_min, bounding_max))
    return bound

def plot_2d(res, label, ca1_label, diameter, resolution, get_color, kde=None, bound=None):

    r = diameter / 2.0
    if bound is None:
        bound = get_bound(res)

    bounding_min, bounding_max = bound
    zs = np.arange(bounding_min[2], bounding_max[2], resolution[2])
    xy_pixels = bound / resolution    

    cell_coords = []

    if kde:
        kde_coords = []
    for z in zs:
        xys = []
        z_min = z - r
        z_max = z + r
        for i, coord in enumerate(res):
            if z_min <= coord[2] <= z_max:
                color = np.hstack((ca1_label[i], label[i]))
                xys.append([coord, color])
        cell_coords.append([[x[0][0] for x in xys], [x[0][1] for x in xys], [get_color(x[1]) for x in xys]])
        if kde:
            pass
    
    fig = plt.figure()
    im_axes = plt.axes([0.025, 0.11, 0.80, 0.90])
    ind_axes = plt.axes([0.05, 0.05, 0.85, 0.03])
    indplot = ind_axes.plot(np.arange(len(zs)), np.zeros(len(zs)))
    cellplot = im_axes.scatter(cell_coords[0][0], cell_coords[0][1], c=cell_coords[0][2], s=r)
    interactive_fig = InteractiveFig(fig, control_axes=[ind_axes], plots=[cellplot], data = [cell_coords])

    plt.show()
    plt.close()



def get_args():
    parser = argparse.ArgumentParser(\
            description = "Analysis file for Yosuke's CellProfiler result",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--target",
            help = "The two experimental cell groups to be analyzed, arc and h1a",
            nargs = 2)
    parser.add_argument("-r", "--reference",
            help = "The reference cell groups to be sampled from")
    parser.add_argument("-d", "--cell-diameter",
            help = "The diameter of rendered cell, in um",
            type = float,
            default = 13)
    parser.add_argument("-3", "--plot-3d",
            help = "plot data in 3d",
            action = "store_true")
    parser.add_argument("-2", "--plot-2d",
            help = "plot the data in 2d stacks",
            action = "store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    res, label = get_cells(args.reference, args.target)
    ca1_label = select_ca1(res)

    print(res.shape)

    plot_data = []

    cell_colors = ['blue', 'green', 'red', 'yellow', 'brown']
    cell_labels = ["DAPI", "Arc", "H1a", "Arc+H1a", 'non-CA1']
    print("DAPI in CA1:", np.sum(ca1_label))
    print("Arc in CA1:", np.sum(label[:,0] & ca1_label))
    print("H1a in CA1:", np.sum(label[:,1] & ca1_label))
    print("Arc/H1a in CA1:", np.sum(label[:,0] & label[:,1] & ca1_label))
    print("DAPI outside CA1:", np.sum(~ca1_label))
    

    res_ca1 = res[ca1_label]
    '''
    kde_dapi = stats.gaussian_kde(res_ca1.T)
    #kde_arc = stats.gaussian_kde((res_ca1[label[:,0]]).T)
    #kde_h1a = stats.gaussian_kde((res_ca1[label[:,1]]).T)
    mesh_resolution = 30
    bound = get_bound(res)
    print(bound.shape, bound)
    x = np.linspace(bound[0,0], bound[1,0], mesh_resolution)
    y = np.linspace(bound[0,1], bound[1,1], mesh_resolution)
    z = np.linspace(bound[0,2], bound[1,2], mesh_resolution)
    mesh = np.meshgrid(x,y,z)
    print([x.shape for x in mesh])
    cx = np.vstack((mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()))
    clist = kde_dapi(cx)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel(), s=15, c=clist, marker=".", edgecolor="none", cmap=alpha_red)
    plt.show()
    quit()
    '''

    if args.plot_3d:
        plot_3d(res, label, ca1_label, cell_colors, cell_labels)
    if args.plot_2d:
        def get_color(arr):
            if not arr[0]:
                return "brown"
            else:
                if arr[1] and not arr[2]:
                    return "red"
                elif arr[1] and arr[2]:
                    return "yellow"
                elif not arr[1] and arr[2]:
                    return "green"
                elif not arr[1] and not arr[2]:
                    return "blue"
            
        plot_2d(res, label, ca1_label, diameter=150, resolution=np.array([0.73,0.73,1]), get_color=get_color)

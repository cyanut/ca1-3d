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
from matplotlib.backend_bases import KeyEvent, MouseEvent
import pickle


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

def get_color(arr):
    if not arr[0]:
        return (0.647, 0.167, 0.167, 1)
    else:
        if arr[1] and not arr[2]:
            return (0, 1, 0, 1)
        elif arr[1] and arr[2]:
            return (1, 1, 0, 1)
        elif not arr[1] and arr[2]:
            return (1, 0, 0, 1)
        elif not arr[1] and not arr[2]:
            return (0, 0, 1, 1)

def plot_2d(res, label, ca1_label, diameter, resolution, get_color, kde=None, bound=None):
    r = diameter / 2.0
    if bound is None:
        bound = get_bound(res)

    bounding_min, bounding_max = bound
    zs = np.arange(bounding_min[2], bounding_max[2], resolution[2])
    xy_pixels = bound / resolution    

    cell_coords = []

    cell_z_index = {}
    z_cell_index= []
    for j, z in enumerate(zs):
        xys = []
        z_min = z - r
        z_max = z + r
        cell_index = []
        for i, coord in enumerate(res):
            if z_min <= coord[2] <= z_max:
                if not i in cell_z_index:
                    cell_z_index[i] = []
                cell_z_index[i].append(j)
                cell_index.append(i)
                color = np.hstack((ca1_label[i], label[i]))
                xys.append([coord, color])
        z_cell_index.append(cell_index)
        cell_coords.append([[x[0][0] for x in xys], [x[0][1] for x in xys], [get_color(x[1]) for x in xys]])
        if kde:
            pass
    
    fig = plt.figure()
    im_axes = plt.axes([0.025, 0.11, 0.80, 0.90])
    ind_axes = plt.axes([0.05, 0.05, 0.85, 0.03])
    indplot = ind_axes.plot(np.arange(len(zs)), np.zeros(len(zs)))
    cellplot = im_axes.scatter(cell_coords[0][0], cell_coords[0][1], c=cell_coords[0][2], s=4*r*r)#here pi=4 :P
    im_axes.set_ylim(bound[:,1] + np.array([-r, r])) 
    im_axes.set_xlim(bound[:,0] + np.array([-r, r])) 

    def click_handler(plot, event):
        seq = [np.array([True, False, False]),
               np.array([True, True, False]),
               np.array([True, False, True]),
               np.array([True, True, True]),
               np.array([False, False, False])]

        if isinstance(event, MouseEvent) and event.inaxes is im_axes:
            x = event.xdata
            y = event.ydata
            z = zs[plot.index]
            
            cell_id = None
            min_dist = 99999999999
            for i, coord in enumerate(res):
                if z - r <= coord[2] <= z + r:
                    dist = (x - coord[0]) ** 2 + (y - coord[1]) ** 2
                    if dist < min_dist:
                        cell_id = i
                        min_dist = dist
            if cell_id:
                label_arr = np.hstack((ca1_label[cell_id], label[cell_id]))
                i = 0
                while not np.all(label_arr == seq[i]):
                    i += 1
                if event.button == 1:
                    i += 1
                else:
                    i -= 1
                i = i % len(seq)
                label[cell_id] = seq[i][1:]
                ca1_label[cell_id] = seq[i][0]
                #TODO
                #update plot data for all affected z
                #find affected z-stack
                affected_z = cell_z_index[cell_id]
                new_color = get_color(seq[i])
                for z in affected_z:
                    z = int(z)
                    #find cell in z
                    cell_num = z_cell_index[z].index(cell_id)
                    #update plot data
                    cell_coords[z][2][cell_num] = new_color

                #update plot
                cellplot.set_facecolor(cell_coords[plot.index][2])
                fig.canvas.draw()
                

    interactive_fig = InteractiveFig(fig, control_axes=[ind_axes], plots=[cellplot], data = [cell_coords], event_handler=click_handler)

    plt.show()
    plt.close()



def get_args():
    parser = argparse.ArgumentParser(\
            description = "Analysis file for Yosuke's CellProfiler result",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data",
            help = "Load data file (pickled)",
            nargs = "?")
    parser.add_argument("-o", "--output",
            help = "Output dump")
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
    if args.data:
        with open(args.data,"rb") as fdata:
            data = pickle.load(fdata)
        res = data["res"]
        label = data["label"]
        ca1_label = data["ca1_label"]
    else:
        res, label = get_cells(args.reference, args.target)
        ca1_label = select_ca1(res)
        


    plot_data = []

    cell_colors = ['blue', 'green', 'red', 'yellow', 'brown']
    cell_labels = ["DAPI", "Arc", "H1a", "Arc+H1a", 'non-CA1']
    

    res_ca1 = res[ca1_label]
    
    print("DAPI in CA1:", np.sum(ca1_label))
    print("Arc in CA1:", np.sum(label[:,0] & ca1_label))
    print("H1a in CA1:", np.sum(label[:,1] & ca1_label))
    print("Arc/H1a in CA1:", np.sum(label[:,0] & label[:,1] & ca1_label))
    print("DAPI outside CA1:", np.sum(~ca1_label))

    if args.plot_3d:
        plot_3d(res, label, ca1_label, cell_colors, cell_labels)
    if args.plot_2d:
            
        plot_2d(res, label, ca1_label, diameter=15, resolution=np.array([0.73,0.73,1]), get_color=get_color)
    
    if args.output:
        with open(args.output, 'wb') as outf:
            pickle.dump({"res":res, "label":label, "ca1_label":ca1_label}, outf)
            
    print("DAPI in CA1:", np.sum(ca1_label))
    print("Arc in CA1:", np.sum(label[:,0] & ca1_label))
    print("H1a in CA1:", np.sum(label[:,1] & ca1_label))
    print("Arc/H1a in CA1:", np.sum(label[:,0] & label[:,1] & ca1_label))
    print("DAPI outside CA1:", np.sum(~ca1_label))
    print("------------------")
    
    
    kde_dapi = stats.gaussian_kde(res_ca1.T)
    kde_arc = stats.gaussian_kde((res[label[:,0] & ca1_label]).T)
    kde_h1a = stats.gaussian_kde((res[label[:,1] & ca1_label]).T)
    cells = res[ca1_label]
    dapi_val = kde_dapi(cells.T)
    arc_val = kde_arc(cells.T)
    h1a_val = kde_h1a(cells.T)
    kld_da = stats.entropy(dapi_val, arc_val)
    kld_dh = stats.entropy(dapi_val, h1a_val)
    kld_ah = stats.entropy(arc_val, h1a_val)
    print("KL divergence between DAPI KDE and arc KDE:", kld_da)
    print("KL divergence between DAPI KDE and h1a KDE:", kld_dh)
    print("KL divergence between arc KDE and h1a KDE:", kld_ah)

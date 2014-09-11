#!/usr/bin/python3
from __future__ import (absolute_import, division, print_function, unicode_literals)
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist


def load_txt(fname):
    ''' 
    Load CellProfiler text file as dictionaries. 
    fname: file path of the file
    return: dictionary with titles as keys and list of values as values.
    '''
    f = open(fname).readlines()
    title = f[0].strip().split("\t")
    dic = {}
    for t in title:
        dic[t] = []
    for line in f[1:]:
        l = line.strip().split("\t")
        for t, n in zip(title, l):
            dic[t].append(n)
    return dic

def get_track_objects(res_dic, mask=None, pixel_size=(0.332, 0.332, 1.0)):
    '''
    Get relavant information (XYZ Coordinates, track labels from 
    dictionary returned by load_txt and sanitize input by 
    removing nans, and calculate centre coordinate of the cells.
    res_dic: input dictionary
    mask: numpy array of mask image, ROI = 1, outside = 0
    pixel_size: actual size of image pixels in um. A tuple of (X, Y, Z).
    return: a numpy array of shape (cell number, 3) where each row 
            is the XYZ coordinate of a cell.
    '''
    cell_info_raw = zip(res_dic["TrackObjects_Label_1"],
                    res_dic["Location_Center_X"],
                    res_dic["Location_Center_Y"],
                    res_dic["ImageNumber"]
                    )
    max_n = 0

    cell_info = {}
    for cell in cell_info_raw:


        try:
            n = int(float(cell[0]))
        except ValueError:
            continue

        if mask:
            if mask[cell_info[0], cell_info[1]] == 0:
                continue

        if not n in cell_info:
            cell_info[n] = [[],[],[]]
        
        cell_info[n][0].append(float(cell[1])*pixel_size[0])#X
        cell_info[n][1].append(float(cell[2])*pixel_size[1])#Y
        cell_info[n][2].append(float(cell[3])*pixel_size[2])#Z
    
    cell_info_avg = {}
    for cell,v in cell_info.items():
        cell_info_avg[cell] = [sum(x) / len(x) for x in v]

    return np.array(list(cell_info_avg.values()))

def get_distance_distro(tracked_objects, sample_size=None, repeat=1):
    '''
    Given an 2d array of coordinates, random sample from it calculate pair-wise distances.
    tracked_objects: input 2d array. Each row is a coordinate.
    sample_size: the size of random sample to be withdrawn, and if is None,
                 calculate pair-wise distance of the whole input.
    repeat: number of random samples to be drawn.
    return: a 1d array of distances, pooled from all samples.
    '''

    if sample_size is None:
        sample_size = tracked_objects.shape[0]
    dist = []
    ind_array = np.arange(tracked_objects.shape[0])
    for i in range(repeat):
        np.random.shuffle(ind_array)
        selected_objects = tracked_objects[ind_array[:sample_size],:]
        dist.append(pdist(selected_objects))

    dist = np.hstack(dist)

    return dist


def get_args():
    '''
    Commandline argument parsing.
    '''
    parser = argparse.ArgumentParser(\
            description = "Analysis file for Yosuke's CellProfiler result",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--target",
            help = "The experimental cell groups to be analyzed",
            nargs = "+")
    parser.add_argument("-r", "--reference",
            help = "The reference cell groups to be sampled from",
            nargs = "*")
    parser.add_argument("-m", "--mask",
            help = "A mask image where the value inside region of interest is 1 and outside is 0, NOT FULLY IMPLEMENTED")
    parser.add_argument("-b", "--bins",
            help = "Number of bins to be used in the histogram",
            type = int,
            default = 50)
    parser.add_argument("-rb", "--reference-bins",
            help = "Number of bins to be used in the histogram of refernece distribution",
            type = int,
            default = 50)
    parser.add_argument("-n", "--repeat",
            help = "Number of random samples to be drawn from reference cell groups",
            type = int,
            default = 100)
    parser.add_argument("-c", "--cumulative",
            help = "plot cumulative histogram",
            action="store_true")
    parser.add_argument("--hist-type",
            help = "Type of histogram",
            choices = ["step", "bar"],
            default = "bar")
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    args = get_args()
    targets = args.target
    references = args.reference
    assert len(targets) == len(references), \
        "Number of target cell groups have to equal number of reference cell groups."

    bins = args.bins
    repeat = args.repeat

    pooled_target_dist = np.array([])
    pooled_reference_dist = np.array([])

    for target, reference in zip(targets, references):
        target_info = load_txt(target)
        reference_info = load_txt(reference)
        
        target_objects = get_track_objects(target_info, mask = args.mask)
        reference_objects = get_track_objects(reference_info, mask = args.mask)

        print("Loaded {} cells from target file: {}.".format(target_objects.shape[0], target))
        print("Loaded {} cells from reference file: {}.".format(reference_objects.shape[0], reference))
    
        target_dist = get_distance_distro(target_objects)
        reference_dist = get_distance_distro(\
                    reference_objects,
                    sample_size = target_objects.shape[0],
                    repeat = repeat
                    )

        pooled_target_dist = np.hstack(target_dist)
        pooled_reference_dist = np.hstack(reference_dist)


    print("plotting reference distribution:", pooled_reference_dist.shape)
    plt.hist(reference_dist, bins=args.reference_bins, histtype=args.hist_type, cumulative=args.cumulative,normed=True)
    print("plotting target distribution:", pooled_target_dist.shape)
    plt.hist(target_dist, bins=bins, histtype=args.hist_type, normed=True, cumulative=args.cumulative)

    print("Kolmogorov-Smirnof test: target against reference (two tailed)")
    D, p = ks_2samp(pooled_target_dist, pooled_reference_dist)
    print("D = {}, p = {}".format(D, p))
    

    plt.show()

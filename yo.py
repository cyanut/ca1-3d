#!/usr/bin/python3
from __future__ import (absolute_import, division, print_function, unicode_literals)
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist, squareform


def load_txt(fname):
    ''' 
    Load CellProfiler text file as dictionaries. 
    fname: file path of the file
    return: dictionary with titles as keys and list of values as values.
    '''
    fi = open(fname)
    f = fi.readlines()
    fi.close()
    title = f[0].strip().split("\t")
    dic = {}
    for t in title:
        dic[t] = []
    for line in f[1:]:
        l = line.strip().split("\t")
        for t, n in zip(title, l):
            dic[t].append(n)

    return dic

def get_cell_info_raw(res_dic):
    cell_info_raw = []
    for label,x,y,z in zip(res_dic["TrackObjects_Label_1"],
                    res_dic["Location_Center_X"],
                    res_dic["Location_Center_Y"],
                    res_dic["ImageNumber"]
                    ):
        try:
            n = int(float(label))
        except ValueError:
            continue
        cell_info_raw.append([int(float((label))), float(x), float(y), float(z)])
    return cell_info_raw


def get_colocal_arrays(reference, target_list, tol=1.0, pixel_size=(0.332, 0.332, 100)):
    '''
    Match target_list objects onto reference. 
    return: a boolean numpy arra
    '''

    tol **= 2
    pixel_size = np.array(pixel_size)[np.newaxis, :]
    ref = np.array(get_cell_info_raw(reference))[:,1:]
    target_list = [np.array(get_cell_info_raw(x))[:, 1:] for x in target_list]
    ref *= pixel_size
    target_list = [x * pixel_size for x in target_list]
    res = (np.zeros((ref.shape[0], len(target_list))) != 0)
    for i,coord in enumerate(ref):
        for j,target in enumerate(target_list):
            #print("target:",target.shape, "coord:", coord.shape)
            #print(np.min(np.sum((target - coord) ** 2, axis=1)))
            res[i,j] = (np.min(np.sum((target - coord) ** 2, axis=1)) < tol)

    return res       
    


def get_track_objects(res_dic, mask=None, pixel_size=(0.332, 0.332, 1.0), colocal_info_array=None):
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
    cell_info_raw = get_cell_info_raw(res_dic)
    max_n = 0

    cell_info = {}
    colocal_info = {}
    for i, cell in enumerate(cell_info_raw):


        n = cell[0]

        if mask:
            if mask[cell_info[0], cell_info[1]] == 0:
                continue

        if not n in cell_info:
            cell_info[n] = [[],[],[]]
            if colocal_info_array is not None:
                colocal_info[n] = (np.zeros(colocal_info_array.shape[1])!=0)
        
        cell_info[n][0].append(float(cell[1])*pixel_size[0])#X
        cell_info[n][1].append(float(cell[2])*pixel_size[1])#Y
        cell_info[n][2].append(float(cell[3])*pixel_size[2])#Z
        if colocal_info_array is not None:
            colocal_info[n] |= colocal_info_array[i] 
    
    cell_info_avg = []
    colocal_info_list = []
    for cell,v in cell_info.items():
        cell_info_avg.append([sum(x) / len(x) for x in v])
        if colocal_info_array is not None:
            colocal_info_list.append(colocal_info[cell])

    return np.array(cell_info_avg), np.array(colocal_info_list) 

def get_distance_distro(tracked_objects, sample_size=None, repeat=1, neighbours=0):
    '''
    Given an 2d array of coordinates, random sample from it calculate pair-wise distances.
    tracked_objects: input 2d array. Each row is a coordinate.
    sample_size: the size of random sample to be withdrawn, and if is None,
                 calculate pair-wise distance of the whole input.
    repeat: number of random samples to be drawn.
    neighbours: number of nearest neighbours to include in the analysis
    return: a 1d array of distances, pooled from all samples.
    '''

    if sample_size is None:
        sample_size = tracked_objects.shape[0]
    dist = []
    ind_array = np.arange(tracked_objects.shape[0])
    for i in range(repeat):
        np.random.shuffle(ind_array)
        selected_objects = tracked_objects[ind_array[:sample_size],:]
        if neighbours <= 0:
            dist.append(pdist(selected_objects))
        else:
            dist_all = squareform(pdist(selected_objects))
            dist_all.partition(neighbours)
            dist_all = dist_all[:,:neighbours+1]
            dist.append(dist_all[dist_all > 0])

    dist = np.hstack(dist)

    return dist


def get_cells(reference, targets):
    ref_txt = load_txt(reference)
    target_txt_list = [load_txt(x) for x in targets]
    colocal = get_colocal_arrays(ref_txt, target_txt_list)
    res, label = get_track_objects(ref_txt, colocal_info_array = colocal)
    return res, label


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
    parser.add_argument("-nb", "--neighbours",
            help = "Number of nearest neighbours to include, a number smaller or equal to 0 means all cells",
            type = int,
            default = 0)
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
    parser.add_argument("--select_ca1",
            help = "Only analyze cells in ca1",
            type = bool,
            default = "True")
    return parser.parse_args()

def show_distance_distro(target_dist, reference_dist, args):
    target_dist_distribution = get_distance_distro(target_dist, args.neighbours)
    reference_dist_distribution = get_distance_distro(\
                reference_dist,
                sample_size = target_dist.shape[0],
                repeat = args.repeat,
                neighbours = args.neighbours
                )

    print("plotting reference distribution:", reference_dist.shape)
    plt.hist(reference_dist_distribution, bins=args.reference_bins, histtype=args.hist_type, cumulative=args.cumulative,normed=True, alpha=0.5)
    print("plotting target distribution:", target_dist.shape)
    plt.hist(target_dist_distribution, bins=args.bins, histtype=args.hist_type, normed=True, cumulative=args.cumulative, alpha=0.5)

    print("Kolmogorov-Smirnof test: target against reference (two tailed)")
    D, p = ks_2samp(target_dist_distribution, reference_dist_distribution)
    print("D = {}, p = {}".format(D, p))

    plt.show()

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

        colocal_array = get_colocal_arrays(reference_info, [target_info])
        
        reference_objects,colocal_mask = get_track_objects(reference_info, colocal_info_array = colocal_array)
        if args.select_ca1:
            from get_ca1 import select_ca1
            is_ca1 = select_ca1(reference_objects)
        else:
            is_ca1 = (np.ones(reference_objects.shape[0]) == 1)

        target_objects = reference_objects[colocal_mask[:,0] & is_ca1]
        reference_objects = reference_objects[is_ca1]
            

        print("Loaded {} cells from target file: {}.".format(target_objects.shape[0], target))
        print("Loaded {} cells from reference file: {}.".format(reference_objects.shape[0], reference))
    
        target_dist = get_distance_distro(target_objects, neighbours=args.neighbours)
        reference_dist = get_distance_distro(\
                    reference_objects,
                    sample_size = target_objects.shape[0],
                    repeat = repeat,
                    neighbours = args.neighbours
                    )

        pooled_target_dist = np.hstack((pooled_target_dist, target_dist))
        pooled_reference_dist = np.hstack((pooled_target_dist, reference_dist))


    print("plotting reference distribution:", pooled_reference_dist.shape)
    plt.hist(pooled_reference_dist, bins=args.reference_bins, histtype=args.hist_type, cumulative=args.cumulative,normed=True, alpha=0.5)
    print("plotting target distribution:", pooled_target_dist.shape)
    plt.hist(pooled_target_dist, bins=bins, histtype=args.hist_type, normed=True, cumulative=args.cumulative, alpha=0.5)

    print("Kolmogorov-Smirnof test: target against reference (two tailed)")
    D, p = ks_2samp(pooled_target_dist, pooled_reference_dist)
    print("D = {}, p = {}".format(D, p))
    

    plt.show()

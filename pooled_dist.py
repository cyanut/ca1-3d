import argparse
from yo import get_distance_distro, ks_2samp, plt
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser(\
            description = "Show distance distribution of pooled counts",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data",
            help = "Load data file (pickled)",
            nargs = "+") 
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

def get_pool_res(data):
    pooled_data = {}
    for k in data[0].keys():
        pooled_data[k] = []
    for d in data:
        for k in pooled_data.keys():
            pooled_data[k].append(d[k])
    for k in pooled_data.keys():
        pooled_data[k] = np.concatenate(pooled_data[k])
    return pooled_data

if __name__ == "__main__":
    args = get_args()
    data = []
    for d in args.data:
        with open(d, "rb") as f:
            data.append(pickle.load(f))

    group_num = data[0]["label"].shape[1]
    pooled_dist = [[]] * group_num
    
    for d in data:

        ref = d["res"][d["ca1_label"]]
        for g in range(group_num):
            target = d["res"][d["label"][:,g] & d["ca1_label"]]
            target_dist = get_distance_distro(target)
            ref_dist = get_distance_distro(ref, sample_size=target.shape[0],
                    repeat = args.repeat)
            pooled_dist[g].append([target_dist, ref_dist])

        
    for g in range(group_num):
        print()
        pooled_dist[g] = [np.hstack(x) for x in zip(*pooled_dist[g])]

        print("plotting reference distribution:", pooled_dist[g][1].shape)
        ref = plt.hist(pooled_dist[g][1], bins=args.reference_bins, histtype=args.hist_type, cumulative=args.cumulative,normed=True, alpha=0.5)
        print("plotting label {} distribution: {}".format(g, pooled_dist[g][0].shape))
        target = plt.hist(pooled_dist[g][0], bins=args.bins, histtype=args.hist_type, normed=True, cumulative=args.cumulative, alpha=0.5)
        print("Kolmogorov-Smirnof test: target against reference (two tailed)")
        D, p = ks_2samp(pooled_dist[g][0], pooled_dist[g][1])
        print("D = {}, p = {}".format(D, p))

        plt.show()
 



import tifffile
import numpy as np

def test_func(i):
    return 200 * (i<=2)

def threshold(img_stack, func):
    n_stack = img_stack.shape[0]
    for i in range(n_stack):
        img_stack[0][img_stack[0] < func(i)] = 0
    return img_stack

if __name__ == "__main__":
    import sys
    fname = sys.argv[1]
    ofname = sys.argv[2]
    img_stack = tifffile.TiffFile(fname).asarray()
    img_stack = threshold(img_stack, test_func)
    tifffile.imsave(ofname, img_stack)




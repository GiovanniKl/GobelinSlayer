"""Script for extracting/creating embroidery patterns for
gobelins/tapisseries from (low-quality) internet images or your own
designs made, e.g., in Windows Paint.

Created by Jan Klíma on 2024/04/30.
Updated on 2024/04/30.
"""

import numpy as np
from PIL import Image
import colorsys
import matplotlib.pyplot as plt  # just for testing
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks  # for color finding
from sklearn.cluster import KMeans  # for color finding

# ### adjustable constants (advanced usage)
# kernel filling factor in units of cellsize, default: {".": 0.45, "+": 0.8}
#   (increase it if your kernel is too small)
FILL = {".": 0.45, "+": 0.8}
# FFT filter params: circle radius (default 5), replaced value (default 10.)
FFT = {"radius": 5, "repval": 10.}


def main():
    """Main function where you set all parameters, and where all the
    code is processed.
    """
    # place to read and save your designs
    folder = (r"C:\Users\Jan\Documents\programování\3-1415926535897932384626"
              + r"433832791thon\gobelinSlayer")
    # name of the (future) saved file (without extension)
    save = "sykorky-test0"
    # image file to read the pattern from (located within 'folder')
    read = "IMG-20240429-WA0003.jpg"
    # float, (px/cell) amount of pixels per one cell/stitch in the source
    #   (accepts floats, since some images can have non-integer cells/side)
    pxpcell = 443/120
    # int, number of colors in the image (nocolors >= 0; 0 means automatic)
    ncolors = 25
    # int, minimum number of grid points for rows/columns for the final grid
    minr, minc = 20, 20
    # int, dots per cell/grid point (affects final resolution)
    dpc = 10
    # bool, print symbols to cells? (for better clarity between colors)
    imprintsym = False
    
    # ### advanced options ###
    # int, grid line thickness for 1x1 cell and 10x10 cell
    tgrid1, tgrid10 = max((1, dpc//10)), max((2, dpc//5))
    # hex color, grid line color for 1x1 cell and 10x10 cell
    cgrid1, cgrid10 = "#888888", "#444444"
    cbg = "#ffffff"  # hex color, background color
    pad = 2  # int, (multiple of cellsize) space around grid and table
    kernel = "."  # string, {".", "+"} defines shape to extract color from
    filter_grid = True  # bool, filter out grid remnants?

    # ##### DO NOT EDIT ANYTHING FURTHER DOWN (ONLY FIXES ALLOWED) #####
    pattern, colors, counts, r, c = from_raw_src(folder+"/"+read, pxpcell,
                                                 ncolors, kernel, filter_grid)


def from_raw_src(src, pxpcell, ncolors, kernel, filter_grid):
    """Reads and processes the input image (characterizes colours and
    makes a color array representation of the pattern).
    - src - string, path to the source image.
    - pxpcell - int, number of pixel per stitch/cell in the source image.
    - ncolors - int, (>=0) number of colours to characterize, 0 means
        automatic recognition.
    - kernel - string, {".", "+"} shape to extract colour from.

    Returns:
    - pattern - 2darray of ints, representation of the pattern,
        each int value represents a color from 'colors'.
    - clrs - list of strings, list of colours, ordered by appearance
        in the source image.
    - cts - list of ints, list of occurences (same size and order as 'clrs').
    - r, c - ints, number of rows and columns in the pattern.
    """
    im = np.asarray(Image.open(src))  # read image and make [r, c, RGB] array
    # <the RGB is a vector of size 3 of ints (0-255))>
    assert im.shape[2] == 3  # check if alpha is not present
    assert ncolors > 0, "Automatic ncolor recognition not yet implemented!"
    if filter_grid:
        im = do_filter_grid(im, pxpcell)
        # im3 = Image.fromarray(im, mode="RGB")
        # im3.save("test3.png", "PNG")
    
    r, c = int(im.shape[0]/pxpcell), int(im.shape[1]/pxpcell)
    kern = get_kernel(kernel, int(round(pxpcell)))
    colors = np.zeros((r, c, 3), dtype=np.uint8)
    for ri in range(r):
        for ci in range(c):
            cs = np.zeros((len(kern), 3), dtype=int)
            for i, ei in enumerate(kern):
                cs[i] = im[ei[0]+int(ri*pxpcell), ei[1]+int(ci*pxpcell)]
            colors[ri, ci] = np.mean(cs, 0, dtype=int)
    # print(r, c, colors.shape)  # , colors/255)

    # im1 = Image.fromarray(colors, mode="RGB")
    # im1.save("test1.png", "PNG")
    
    clrs, inds = get_colors(colors, ncolors)
    pattern = clrs[inds]
    cts = [np.sum(inds == i) for i in range(ncolors)]
    
    # im2 = Image.fromarray(pattern, mode="RGB")
    # im2.save("test2.png", "PNG")
    return pattern, clrs, cts, r, c


def do_filter_grid(im, pxpcell):
    """Executes the remanent grid filtering."""
    poses = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1],
                      [1, 1], [1, 0], [1, -1], [0, -1]], dtype=int)
    imsh = np.array(im.shape[:-1], dtype=int)
    circs = (imsh/2).astype(int) + poses*(imsh/pxpcell).astype(int)
    mask = get_mask(imsh[0], imsh[1], circs, FFT["radius"])
    hsv = rgb2hsv(im/255)
    val = hsv[..., 2]  # extract Value channel (grid is black)

    valfft = np.fft.fftshift(np.fft.fft2(val))*mask
    valfft[mask==0] = FFT["repval"]
    val0 = np.abs(np.fft.ifft2(np.fft.fftshift(valfft)))
    val0[val0 > 1] = 1.  # put all values within [0, 1]
    val0[val0 < 0] = 0.

    hsv[..., 2] = val0
    return (hsv2rgb(hsv)*255).astype("uint8")
    


def get_mask(r, c, rcs, radius):
    """Creates a mask for fft filtering of size r×c with zeros at
    positions specified in rcs in circles with given radius in px.
    """
    mask = np.ones((r, c))
    cs = radius*2+1  # circle size
    circle = np.ones((cs, cs))
    for ri in range(cs):
        for ci in range(cs):
            if (ri-radius)**2 + (ci-radius)**2 <= radius**2:
                circle[ri, ci] = 0
    for ri, ci in rcs:
        # assumes the circle is always fully within the mask
        mask[ri-radius:ri+radius+1, ci-radius:ci+radius+1] *= circle
    return mask


def get_colors(image, ncolors):
    """Determine most common colors in an image (array) using k-means
    clustering (from scikit learn module).
    - image - array of shape [r, c, 3], RGB image array.
    - ncolors - int, number of most common colors to find.
    Returns:
    - colors - array of shape [ncolors, 3], found colors.
    - labels - array of shape [r, c], indices of 'colors'.
    """
    clt = KMeans(n_clusters=ncolors)
    imshape = image.shape[:-1]
    clt.fit(image.reshape(-1, 3))
    return clt.cluster_centers_.astype("uint8"), clt.labels_.reshape(imshape)


def get_kernel(kernel_type, cellsize):
    """Returns an array of indices constructing a kernel of given
    type.  For cellsizes 1 and 2 the kernel is always the whole cell.
    - kernel_type - str, {".", "+"} defines the shape/type
    - cellsize - int, size of a cell in px.
    """
    if cellsize == 1:
        return [[0]]
    if cellsize == 2:
        return [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert kernel_type in FILL.keys(), f"Wrong kernel type: {kernel_type}"
    n = cellsize  # just a shortcut
    ker0 = np.zeros((n, n))
    ker = []
    for ri in range(n):
        for ci in range(n):
            ma = np.max(np.abs((np.array([ri, ci]) - (n-1)/2)/(n-1)))
            mi = np.min(np.abs((np.array([ri, ci]) - (n-1)/2)/(n-1)))
            if kernel_type == "." and ma <= FILL["."]*0.5:
                ker0[ri, ci] = 1
                ker.append([ri, ci])
            elif (kernel_type == "+" and mi <= FILL["+"]/4
                  and ma <= FILL["+"]/2):
                ker0[ri, ci] = 1
                ker.append([ri, ci])
    print("This is your kernel:", ker0, sep="\n")
    # return np.array(ker, dtype=int)
    return ker
    

def clamp(x, *args):
    """Used for validation of rgb tuple elements to be between 0 and 255."""
    return max(0, min(x, 255))

def rgb2hex(rgbtup, *args):
    """Converts rgb tuple to a hex string."""
    return "#{:02x}{:02x}{:02x}".format(*[clamp(i) for i in rgbtup])

def hex2rgb(hexstr, *args):
    """Converts hex string to RGB tuple."""
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))


def rgb2hsv(arr):
    """Implemented from matplotlib.colors.

    Convert an array of float RGB values (in the range [0, 1]) to HSV values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to HSV values in range [0, 1]
    """
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {arr.shape} was found.")

    in_shape = arr.shape
    arr = np.array(
        arr, copy=False,
        dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )
    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = np.ptp(arr, -1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)


def hsv2rgb(hsv):
    """Implemented from matplotlib.colors.

    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {hsv.shape} was found.")

    in_shape = hsv.shape
    hsv = np.array(
        hsv, copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


if __name__ == "__main__":
    main()

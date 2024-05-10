"""Script for extracting/creating embroidery patterns for
gobelins/tapisseries from (low-quality) internet images or your own
designs made, e.g., in Windows Paint.

Created by Jan Klíma on 2024/04/30.
Updated on 2024/05/10.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import webcolors  # for conversion of names and hex codes (CSS3)
from sklearn.cluster import KMeans  # for color finding

# ### adjustable constants (advanced usage)
# kernel filling factor in units of cellsize, default: {".": 0.45, "+": 0.8}
#   (increase it if your kernel is too small)
FILL = {".": 0.45, "+": 0.8}
# FFT filter params: circle radius (default 5), replaced value (default 10.)
FFT = {"radius": 5, "repval": 10.}
# fixing random state of color-recognition, int or None (default: None)
RANDOM_STATE = 22863
FONT2CELL = 0.7  # filling factor of fontsize in a cell (default: 0.6)
# path to used font TTF file (monospaced preferred, default: "consolas.ttf")
#   (the font is not supplied by this package, you have to find a font file
#   that suits you and put the link/path to it here)
TTF = "consolas.ttf"
SYMS = "abcdefghijklmnopqrstuvwxyz0123456789-+/*@#!:.ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# reference names for small/large patterns
SHORT = "created in GobelinSlayer"
LONG = "see github.com/GiovanniKl/GobelinSlayer"


def main():
    """Main function where you set all parameters, and where all the
    code is processed.
    """
    # place to read and save your designs
    folder = (r"C:\Users\Jan\Documents\programování\3-1415926535897932384626"
              + r"433832791thon\gobelinSlayer")
    # name of the (future) saved file (without extension)
    save = "kocka0"+f"_seed{RANDOM_STATE}"
    # image file to read the pattern from (located within 'folder')
    read = "IMG-20240429-WA0003_pxpcell443d120_n25.jpg"
    read = "kocka0_seed22863_pattern.png"
    # float, (px/cell) amount of pixels per one cell/stitch in the source
    #   (accepts floats, since some images can have non-integer cells/side,
    #   but the input image must have the same pxpcell in both directions)
    pxpcell = 443/120
    pxpcell = 1  # for reading from a 1:1 pattern
    # int, number of colors in the image (nocolors >= 0; 0 means all)
    ncolors = 0
    # bool, save just pattern? (dpc=1) (can be used for manual editing of
    #   generated pattern before making the final image on a grid, or
    #   for making previews, since its quite fast)
    save_just_pattern = False
    # int, minimum number of grid points for rows/columns for the final grid
    minr, minc = 20, 20
    # int, dots per cell/grid point (affects final resolution)
    dpc = 10
    # bool, print symbols to cells? (for better clarity between colors)
    imprintsym = True
    
    # ### advanced options ###
    # int, grid line thickness for 1x1 cell and 10x10 cell
    tgrid1, tgrid10 = max((1, dpc//10)), max((2, dpc//5))
    # hex color, grid line color for 1x1 cell and 10x10 cell
    cgrid1, cgrid10 = "#888888", "#444444"
    cbg = "#ffffff"  # hex color, background color
    pad = 2  # int, (multiple of cellsize) space around grid and table
    kernel = "."  # string, {".", "+"} defines shape to extract color from
    filter_grid = True  # bool, filter out grid remnants? (only for pxpcell > 1)
    #   (this may or may not work, sometimes changing random seed helps)

    # ##### DO NOT EDIT ANYTHING FURTHER DOWN (ONLY FIXES ALLOWED) #####
    pattern, colors, counts, r, c = from_raw_src(folder+"/"+read, pxpcell,
                                                 ncolors, kernel, filter_grid,
                                                 save_just_pattern, folder+"/"
                                                 + save)
    if ncolors == 0:
        ncolors = len(colors)  # finally change to real number of colors used
    if save_just_pattern:
        print("Pattern image saved successfully (only the 1:1 scale pattern).")
        return  # skips all following code
    pattern, colors, counts = sort_colors(pattern, colors, counts)
    syms, symcolors = get_syms(ncolors, imprintsym, colors)
    table = get_table(ncolors, dpc, colors, counts, max((c, minc)),
                      max((r, minr)), imprintsym, syms, symcolors)
    image = get_image(dpc, r, c, minr, minc, pad, tgrid1, tgrid10, pattern,
                      colors, table, imprintsym, syms, symcolors, cbg, cgrid1,
                      cgrid10)
    image.save(folder+"/"+save+".png")
    image.close()  # release memory
    print("Image saved successfully.")


def get_image(dpc, r, c, minr, minc, pad, tgrid1, tgrid10, pattern, colors,
              table, imprintsym, syms, symcolors, cbg, cgrid1, cgrid10):
    """Creates the final image. First creates empty canvas of size
    [canvasr, canvasc, 3], then paints all pattern stuff, and pastes the
    table. Returns an Image.
    - dpc - int, (px) dots per cell.
    - r/c - ints, rows/columns of grid.
    - minr/minc - ints, minimum number of rows/columns in the final grid.
    - pad - int, (cells) padding in multiples of dpc.
    - tgrid1/tgrid10 - ints, (px) 1x1/10x10 grid thickness.
    - pattern - 2darray of ints, array of color indices making the pattern.
    - colors - array of ints, list colors of shape [ncolors, 3].
    - table - Image, color table image.
    - imprintsym - bool, write help symbols to color cells?
    - syms - str, line of symbols in the same order as 'colors'.
    - symcolors - array of ints, list of colors for the imprinted
        symbols in the same order as 'colors'.
    - cbg - hex string, background color.
    - cgrid1/cgrid10 - hex strings, 1x1/10x10 grid color.
    """
    c0, r0 = max((c, minc)), max((r, minr))
    canvasc = int(2*pad*dpc+c0*(dpc+tgrid1)+np.ceil(c0/10)*(tgrid10-tgrid1))
    canvasr = int(3*pad*dpc+r0*(dpc+tgrid1)+np.ceil(r0/10)*(tgrid10-tgrid1)
                  + table.height)
    canvas = np.zeros((canvasr, canvasc, 3), dtype=np.uint8)
    canvas[:, :] = hex2rgb(cbg)  # put background color everywhere
    # print(canvas)  # see if line above works
    # draw top and left grid lines
    canvas[pad*dpc:pad*dpc+tgrid10, pad*dpc:pad*dpc+c0*(dpc+tgrid1)+tgrid10
           + int(np.ceil(c0/10))*(tgrid10-tgrid1)] = hex2rgb(cgrid10)
    canvas[pad*dpc:pad*dpc+r0*(dpc+tgrid1)+int(np.ceil(r0/10))*(tgrid10-tgrid1)
           + tgrid10, pad*dpc:pad*dpc+tgrid10] = hex2rgb(cgrid10)
    
    canvas = Image.fromarray(canvas)
    ari, aci = pad*dpc+tgrid10, pad*dpc+tgrid10  # initial row/col index of cell
    fontsize = int(FONT2CELL*dpc)
    anchor, dpcd2 = "mm", dpc//2
    font = ImageFont.truetype(TTF, size=fontsize)
    for ri in range(r0):
        for ci in range(c0):
            rl = tgrid10 if ((ri+1) % 10 == 0 or ri == r0-1) else tgrid1
            cl = tgrid10 if ((ci+1) % 10 == 0 or ci == c0-1) else tgrid1
            cgridr = cgrid10 if ((ri+1) % 10 == 0 or ri == r0-1) else cgrid1
            cgridc = cgrid10 if ((ci+1) % 10 == 0 or ci == c0-1) else cgrid1
            arf, acf = ari + dpc, aci + dpc
            if ri < r and ci < c:
                canvas.paste(tuple(colors[pattern[ri, ci]]),
                             (aci, ari, acf, arf))
                if imprintsym:
                    dim = ImageDraw.Draw(canvas)
                    dim.text((aci+dpcd2, ari+dpcd2), syms[pattern[ri, ci]],
                             fill=tuple(symcolors[pattern[ri, ci]]),
                             anchor=anchor, font=font)
            if (ci+1) % 10 == 0 or ci == c0-1:
                canvas.paste(cgridr, (aci, arf, acf+cl, arf+rl))
                canvas.paste(cgridc, (acf, ari, acf+cl, arf+rl))
            else:
                canvas.paste(cgridc, (acf, ari, acf+cl, arf+rl))
                canvas.paste(cgridr, (aci, arf, acf+cl, arf+rl))
            aci = acf+cl
        ari, aci = arf+rl, pad*dpc+tgrid10
    canvas.paste(table, (pad*dpc, 2*pad*dpc+r0*(dpc+tgrid1)
                         + int(np.ceil(r0/10))*(tgrid10-tgrid1)))
    return canvas


def get_table(ncolors, dpc, colors, counts, c, r, imprintsym, syms, symcolors):
    """Creates an image of the color table.
    - ncolors - int, number of colors in the pattern.
    - dpc - int, (px) dots per cell.
    - colors - array of ints, list colors of shape [ncolors, 3].
    - counts - array of ints, list color occurences of shape [ncolors].
    - c/r - int, (cells) number of columns/rows in the pattern.
    - imprintsym - bool, write help symbols to color cells?
    - syms - str, line of symbols in the same order as 'colors'.
    - symcolors - array of ints, list of colors for the imprinted
        symbols in the same order as 'colors'.
    """
    fontsize = int(FONT2CELL*dpc*2)
    anchor, dpcd2 = "lm", dpc//2
    font = ImageFont.truetype(TTF, size=fontsize)
    dummyitem = "#fafad2 lightgoldenrodyellow (1000x)"  # longest assumed line
    linebbox = font.getbbox(dummyitem)
    titler = dpc*3  # height of title cell (1 cell across whole table)
    itemc = (dpc*3+linebbox[2]+1+dpcd2)  # width of an item box
    itemr = dpc*3  # height of an item box
    # number of table columns that fit within pattern width
    #   (approximated, assumes all gridlines are 1 px wide)
    nc = c*(dpc+1)//itemc
    nr = int(np.ceil(ncolors/nc))  # corresponding number of rows
    table = np.zeros((nr*itemr - dpc + titler, nc*itemc, 3), dtype=np.uint8)-1
    im = Image.fromarray(table, mode="RGB")

    for ri in range(nr):
        for ci in range(nc):
            coi = ri*nc+ci  # color index
            if coi >= ncolors:
                continue  # all colors displayed already
            im.paste(tuple(colors[coi]),
                     box=(itemc*ci, itemr*ri+titler,
                          itemc*ci+dpc*2, itemr*ri+dpc*2+titler))
            dim = ImageDraw.Draw(im)
            if imprintsym:
                dim.text((itemc*ci+dpc, itemr*ri+titler+dpc), syms[coi],
                         fill=tuple(symcolors[coi]), anchor="mm", font=font)
            try:
                fontname = webcolors.CSS3_NAMES_TO_HEX[tuple(colors[coi])]
            except KeyError:
                fontname = "<no webname>"
            dim.text((itemc*ci+dpc*2+dpcd2, itemr*ri+dpc+titler),
                     rgb2hex(colors[coi])+" "+fontname+f" ({counts[coi]}x)",
                     fill=(0, 0, 0), anchor=anchor, font=font)
    title = f"{c}x{r}, {ncolors} colors:"
    dim.text((dpc, dpc), title, fill=(0, 0, 0), anchor="lm", font=font)
    if font.getlength(SHORT+LONG+title+"_"*5) < table.shape[1]:
        thanks = f"({SHORT}, {LONG})"
    else:
        thanks = f"({SHORT})"
    dim.text((table.shape[1]-dpc, dpc), thanks, fill=(0, 0, 0), anchor="rm",
             font=font)
    # im.show()
    return im


def get_syms(ncolors, imprintsym, colors):
    """Create list of symbols and their colors if needed.
    - ncolors - int, number of colors in the pattern.
    - imprintsym - bool, write help symbols to color cells?
    - colors - array of ints, list colors of shape [ncolors, 3].
    Returns:
    syms - string, line of symbols to use.
    symcolors - array of ints, list of colors corresponding to each
        symbol.
    """
    if not imprintsym:
        return "", []
    syms = SYMS[:ncolors]
    symcolors = np.zeros((ncolors, 3), dtype=np.uint8)
    for i in range(len(syms)):
        if rgb2hsv(colors[i]/255)[2] < 0.55:
            symcolors[i] = [255, 255, 255]
        else:
            symcolors[i] = [0, 0, 0]
    return syms, symcolors


def sort_colors(pattern, colors, counts):
    """Sorts the colors and updates accordingly the pattern and counts."""
    key = np.flip(np.argsort(counts))  # sort highest to lowest occurence
    colors0, counts0 = colors[key], counts[key]
    pattern0 = pattern.copy()
    for i in range(len(counts)):
        pattern0[pattern == i] = np.nonzero(key == i)[0][0]
    return pattern0, colors0, counts0


def from_raw_src(src, pxpcell, ncolors, kernel, filter_grid, save_just_pattern,
                 savesrc):
    """Reads and processes the input image (characterizes colours and
    makes a color array representation of the pattern).
    - src - string, path to the source image.
    - pxpcell - int, number of pixel per stitch/cell in the source image.
    - ncolors - int, (>=0) number of colours to characterize, 0 means
        automatic recognition.
    - kernel - string, {".", "+"} shape to extract colour from.
    - filter_grid - bool, apply FFT filter to suppress grid?
    - save_just_pattern - bool, save pattern as image with 1:1 scale?
    - savesrc - string, path to the saved image (without extension).

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
    if filter_grid and pxpcell > 1:
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
    if ncolors == 0:
        ncolors = len(clrs)
    cts = [np.sum(inds == i) for i in range(ncolors)]
    # print(clrs, inds, cts, sep="\n")

    if save_just_pattern:
        im2 = Image.fromarray(clrs[inds], mode="RGB")
        im2.save(savesrc+"_pattern.png")
    return inds, clrs, np.array(cts), r, c


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
    - ncolors - int, number of most common colors to find. Can be 0,
        then all colours are used.
    Returns:
    - colors - array of shape [ncolors, 3], found colors.
    - labels - array of shape [r, c], indices of 'colors'.
    """
    shape = image.shape[:-1]  # original shape of the image
    if  ncolors == 0:
        colors, labels = np.unique(image.reshape(-1, 3), axis=0,
                                   return_inverse=True)
        return colors, labels.reshape(shape)
    else:
        clt = KMeans(n_clusters=ncolors, random_state=RANDOM_STATE)
        clt.fit(image.reshape(-1, 3))
        return clt.cluster_centers_.astype("uint8"), clt.labels_.reshape(shape)


def get_kernel(kernel_type, cellsize):
    """Returns an array of indices constructing a kernel of given
    type.  For cellsizes 1 and 2 the kernel is always the whole cell.
    - kernel_type - str, {".", "+"} defines the shape/type
    - cellsize - int, size of a cell in px.
    """
    if cellsize == 1:
        return [[0, 0]]
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

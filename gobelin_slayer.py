"""Script for extracting/creating embroidery patterns for
gobelins/tapisseries from (low-quality) internet images or your own
designs made, e.g., in Windows Paint.

Created by Jan Klíma on 2024/04/30.
Updated on 2024/04/30.
"""

import numpy as np
from PIL import image


def main():
    """Main function where you set all parameters, and where all the
    code is processed.
    """
    # place to read and save your designs
    folder = (r"C:\Users\Jan\Documents\programování\3-1415926535897932384626"
              + r"433832791thon\gobelinSlayer")
    # name of the (future) saved file (without extension)
    save = "sykorky-test0"
    # image file to read the pattern from
    read = "IMG-20240429-WA0003.jpg"
    # int, (px/cell) amount of pixels per one cell/stitch
    pxpcell = 4
    # int, number of colors in the image (nocolors >= 0; 0 means automatic)
    ncolors = 12
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
    cbg = "#dddddd"  # hex color, background color
    pad = 2  # int, (multiple of cellsize) space around grid and table
    kernel = "."  # string, {".", "+"} defines shape to extract color from


if __name__ == "__main__":
    main()

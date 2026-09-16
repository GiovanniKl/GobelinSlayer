# GobelinSlayer
Python program for extraction/creation of tapestry/gobelin patterns from low-quality internet images.

By low quality, I mean images not ready to be used for the cross-stitch embroidery, which includes also images with large resolution but maybe too many colors.

> [!WARNING]
> Currently, this repo is in a very testing state, so I will not make it into a module yet.

## Requirements
I did not do any rigorous testing yet, everything was developed in Python 3.12 and the package versions available at the time of making.

The required packages are (pip install name/import name):
- numpy
- pillow/PIL
- webcolors
- scikit-learn/sklearn
- scikit-image/skimage

## Usage
Currently, the only way to use it is to edit and run the script. I tried to document as much of the things as possible, but it is not very readable — sorry for that. So my recommendation is go through the script and try to get to understand it.


## Outlook
Here are some of the larger changes that I'd like to implement. More things that need attention/fixing can be found in the [Issues](https://github.com/GiovanniKl/GobelinSlayer/issues).
- Clean it up and make it into a repo with a nice user guide and documentation.
- Create a GUI to make it more accessible to non-programmers (including tooltips).
- Fix the FFT filter for grids (as grids are usually periodic, the filter in k-space should also be periodic, not just a few blocking points).



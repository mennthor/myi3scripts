# coding: utf8

"""
Additonal script to `make_scan_map.py` to save some time.
If lnLLH maps were created without smoothing, this script only applies smoothing
to the normal space version of the input map and normalizes it to a PDF.

Find pass2 scan files at:
    /data/ana/Diffuse/HESE/Pass2_reconstruction/reconstruction_tracks
"""
from __future__ import print_function, division

import os
import argparse
import json
import cPickle as pickle
import numpy as np
import healpy as hp


def smooth_type(x):
    """
    Simple wrapper to constrain argparse float to positive values.
    From: stackoverflow.com/questions/12116685
    """
    x = float(x)
    if x < 0.:
        raise argparse.ArgumentTypeError("`smooth` can be in range [0, *].")
    return x


parser = argparse.ArgumentParser(description="HESE scan to healpy map array.")
parser.add_argument("file", type=str, help=("File containing the map. " +
                    "Can be a dict with key 'map' or npy."))
parser.add_argument("--outf", type=str, default="./pdf_map.npy",
                    help="Output file. Default: pdf_map.npy")
parser.add_argument("--outfmt", type=str, choices=["json", "npy", "pickle"],
                    default="npy",
                    help=("Output format: ['json'|'npy'|'pickle']. If 'npy' " +
                          "'pickle' the map array is saved as a numpy array " +
                          "dump. Else a JSON with key 'map' is stored. " +
                          "Default: 'npy'."))
parser.add_argument("--smooth", type=smooth_type, default=0.,
                    help="If given and `>0`, the final map is converted to " +
                    "normal space LLH, smoothed with a gaussian kernel of " +
                    "the  given size in degree and normalized so that the " +
                    "integral over the unit sphere is one. Default: 0.")
args = parser.parse_args()

f = os.path.abspath(os.path.expandvars(os.path.expanduser(args.file)))
outf = os.path.abspath(os.path.expandvars(os.path.expanduser(args.outf)))
outfmt = args.outfmt
smooth = args.smooth

_, ext = os.path.splitext(f)
ext = ext.lower()
if ext == ".npy":
    logl_map = np.atleast_1d(np.load(f))
elif ext == ".json":
    dct = json.load(open(f))
    logl_map = np.atleast_1d(dct["map"])
elif ext == ".pickle":
    dct = pickle.load(open(f))
    logl_map = np.atleast_1d(dct["map"])
else:
    try:
        logl_map = np.atleast_1d(pickle.load(open(f)))
    except Exception:
        raise ValueError("Input file extension is not one of ['npy', 'json', " +
                         "'pickle'] and simply unpickling failed.")

# Get map info
NSIDE = hp.get_nside(logl_map)
print("Loaded map from:\n  {}".format(f))
print("  Resolution is {}".format(NSIDE))

# Normalize to sane values in [*, 0] for conversion llh = exp(logllh)
pdf_map = np.exp(logl_map - np.amax(logl_map))
del logl_map

# Smooth with a gaussian kernel
if smooth > 0.:
    print(u"Smoothing the PDF map with a {:.2f}° kernel".format(smooth))
    pdf_map = hp.smoothing(map_in=pdf_map, sigma=np.deg2rad(smooth),
                           verbose=False)
    # Healpy smoothing is prone to numerical erros, so fix them after smoothing
    pdf_map[pdf_map < 0.] = 0.

# Normalize to PDF, integral is the sum over discrete pixels here
print("Normalizing map to normal space PDF over the unit sphere.")
dA = hp.nside2pixarea(NSIDE)
norm = dA * np.sum(pdf_map)
if norm > 0.:
    pdf_map = pdf_map / norm
    assert np.isclose(np.sum(pdf_map) * dA, 1.)
else:
    print("  !! Map norm is < 0. Returning unnormed map instead !!")

if outfmt == "npy":
    fname = outf if outf.endswith(".npy") else outf + "." + outfmt
    np.save(fname, np.array(pdf_map))
else:
    if outfmt == "json":
        # Takes significantly more space, but is human readable and portable
        out_dict = {"map": list(pdf_map)}  # ndarray can't be JSON serialized
        fname = outf if outf.endswith(".json") else outf + "." + outfmt
        json.dump(out_dict, fp=open(fname, "w"), indent=1,
                  separators=(",", ":"))
    else:
        fname = outf if outf.endswith(".pickle") else outf + "." + outfmt
        pickle.dump(pdf_map, open(fname, "w"))

print("Done. Saved map to:\n  {}".format(fname))

# coding: utf8

"""
Make complete healpy logLLH maps from HESE scans and save in npy or json as
simple arrays.
Find pass2 scan files at:
    /data/ana/Diffuse/HESE/Pass2_reconstruction/reconstruction_tracks

Coordinates are mapped as follows:
- If --coord='equ': ``dec=pi/2-theta`` and ``phi=ra``.
- If --coord='local': ``zen=theta`` and ``azi=phi``.

If `--outfmt` is `json` or `pickle` then further information is saved in a
dictionary with keys:
- 'run_id' : Run ID of the event.
- 'event_id' : Event ID of the event.
- 'mjd' : MJD start tim of the event
- 'NSIDES' : List of combined scanned resolutions.
- 'pixels' : List of scanned pixels.
- 'bf_loc' : Dictionary with keys:
    + 'azi' : Azimuth of best fit pixel, in radians.
    + 'zen' : Zenith of best fit pixel, in radians.
    + 'pix' : Pixel ID of the best fit for the highest included resolution.
- 'bf_equ' : Only if '--coord' is `equ`. Dictionary with keys:
    + 'ra' : Right-ascension of best fit pixel, in radians.
    + 'dec' : Declination of best fit pixel, in radians.
    + 'pix' : Best fit pixel ID for the map transformed to equatorial coords.
"""
from __future__ import print_function, division

import os
import argparse
import re
import json
import cPickle as pickle
from glob import glob
import numpy as np
import healpy as hp

from icecube import dataclasses, millipede, astro
from icecube.dataio import I3File


def smooth_type(x):
    """
    Simple wrapper to constrain argparse float to positive values.
    From: stackoverflow.com/questions/12116685
    """
    x = float(x)
    if x < 0.:
        raise argparse.ArgumentTypeError("`smooth` can be in range [0, *].")
    return x


def rotate_to_equ_pix(NSIDE, mjd, pix):
    """
    Uses a set of pixel IDs for a given resolution NSIDE and transforms local
    to equatorial coordinates. Then returns the new map indices where now the
    map coordinates are ``dec=pi/2-theta`` and ``phi=ra``.
    We can then just use the map as is for equatorial coords by simply
    tranforming ``dec=pi/2-theta``.

    Note: This introduces an error in the order of a pixel width, if the
    transformed new coordinates don't fall on exact pixel centers. For higher
    resolutions this is negliable. The advantage is, that we don't have to
    convert the coordinates every time we want use the equatorial map, but need
    only to do the simple ``dec = pi / 2 - theta`` transformation.

    Parameters
    ----------
    NSIDE : int
        Healpy map resolution.
    mjd : float
        Modified Julian date of the event needed to transform coordinates.
    pix : array-like, int
        Pixel indices to transform to new coordinates.

    Returns
    -------
    pix : array-like
        New pixel indices, so that ``dec=pi/2-theta`` and ``phi=ra``.
    """
    pix = np.atleast_1d(pix)
    npix = len(np.unique(pix))
    th, phi = hp.pix2ang(NSIDE, pix)
    ra, dec = astro.dir_to_equa(zenith=th, azimuth=phi, mjd=mjd)
    # Convert back to healpy map coordinate ranges and get new pixel indices
    th, phi = np.pi / 2. - dec, ra
    pix = hp.ang2pix(NSIDE, theta=th, phi=phi)
    if len(np.unique(pix)) != npix:
        raise RuntimeError("Rotation is not bivariate, possibly due to a " +
                           "too coarse pixelization.")
    return pix


def smooth_and_norm(logl_map, smooth_sigma):
    """
    Takes a lnLLH map, converts it to normal space, applies gaussian smoothing
    and normalizes it, so that the integral over the unit sphere is 1.

    Parameters
    ----------
    logl_map : array-like
        Valied healpy map with lnLLH values.
    smooth : float
        Width in sigma of gaussian smoothing kernel, must be ``>0.``.

    Returns
    -------
    pdf_map : array-like
        Smoothed and normed normal space map. Can be used as a spatial PDF.
    """
    # Normalize to sane values in [*, 0] for conversion llh = exp(logllh)
    pdf_map = np.exp(logl_map - np.amax(logl_map))

    # Smooth with a gaussian kernel
    pdf_map = hp.smoothing(map_in=pdf_map, sigma=smooth_sigma, verbose=False)
    # Healpy smoothing is prone to numerical erros, so fix them after smoothing
    pdf_map[pdf_map < 0.] = 0.

    # Normalize to PDF, integral is the sum over discrete pixels here
    NSIDE = hp.get_nside(logl_map)
    dA = hp.nside2pixarea(NSIDE)
    norm = dA * np.sum(pdf_map)
    if norm > 0.:
        pdf_map = pdf_map / norm
        assert np.isclose(np.sum(pdf_map) * dA, 1.)
    else:
        print("  !! Map norm is < 0. Returning unnormed map instead !!")

    return pdf_map


parser = argparse.ArgumentParser(description="HESE scan to healpy map array.")
parser.add_argument("folder", type=str,
                    help="Folder which contains the scan i3 files")
parser.add_argument("--outf", type=str, default="./{:d}_{:d}_scan_map.npy",
                    help=("Output file. Default: " +
                          "'./<run_id>_<event_id>_scan_map'."))
parser.add_argument("--outfmt", type=str, choices=["json", "npy", "pickle"],
                    default="npy",
                    help=("Output format: ['json'|'npy'|'pickle']. If 'npy' " +
                          "only the logl map array is saved. Else a dict " +
                          "with more info is stored. Default: 'npy'."))
parser.add_argument("--coord", type=str, choices=["equ", "local"],
                    default="local", help="Used coordinate mapping with " +
                    "respect to healpy (theta, phi) coords. Default: 'local'.")
parser.add_argument("--smooth", type=smooth_type, default=0.,
                    help="If given and `>0`, the final map is converted to " +
                    "normal space LLH, smoothed with a gaussian kernel of " +
                    "the  given size in degree and normalized so that the " +
                    "integral over the unit sphere is one. Default: 0.")
args = parser.parse_args()

folder = os.path.abspath(args.folder)
outf = os.path.abspath(os.path.expandvars(os.path.expanduser(args.outf)))
outfmt = args.outfmt
coord = args.coord
smooth = args.smooth

# Fixed here
icemodel = "SpiceMie"

# Make input file list, start at the lowest resolution
srch_str = os.path.join(folder, icemodel + "_nside????.i3.bz2")
f = sorted(glob(srch_str))
fnames = map(os.path.basename, f)
NSIDES = map(lambda s: int(re.search("nside([0-9]{4})", s).group(1)), fnames)

ev_header = {"run_id": None, "event_id": None, "mjd": None}
map_info = {"NSIDES": [], "pixels": []}

# Start with the lowest possible resolution
logl_map = np.zeros(hp.nside2npix(1), dtype=float)

# This is a switch used, if map coord transformation fails for coarse
# resolutions. For speed reasons, pixels are transformed for the scanned pix
# only. If this fails (eg. NSIDE=8) then in the next step all pixels are tried
# to transform, which takes usually longer but should work then.
trafo_next = False

# Loop through files and create the healpy array. The map is build up to the
# higest scan resolution. All others are scaled up pixel-wise
print("")
print("Input folder is:\n  {}".format(folder))
for i, fi in enumerate(f):
    NSIDE = NSIDES[i]
    print("Working on file: {}".format(fnames[i]))
    print("  Resolution is: {}".format(NSIDE))

    # Open the i3 file and get logl info at the pixels
    i3f = I3File(fi)
    # Skip first summary P frame (duplicated best fit frame)
    frame = i3f.pop_physics()

    # Test if files belong to same event
    run_id = frame["I3EventHeader"].run_id
    event_id = frame["I3EventHeader"].event_id
    mjd = frame["I3EventHeader"].start_time.mod_julian_day_double
    if np.all(map(lambda v: v is not None, ev_header.values())):
        assert run_id == ev_header["run_id"]
        assert event_id == ev_header["event_id"]
        assert mjd == ev_header["mjd"]
    else:  # Create on first file
        ev_header["run_id"] = run_id
        ev_header["event_id"] = event_id
        ev_header["mjd"] = mjd

    # Update maps
    map_i, pix_ids = [], []
    while i3f.more():
        frame = i3f.pop_physics()
        assert frame["HealpixNSide"].value == NSIDE

        # Save millipede +lnLLH value in the temp map
        pix_id = frame["HealpixPixel"].value
        map_i.append(-frame["MillipedeStarting2ndPassFitParams"].logl)
        pix_ids.append(pix_id)
    i3f.close()

    # Update verified map info
    map_info["NSIDES"].append(NSIDE)
    map_info["pixels"].append(pix_ids)

    # Set +-inf or nans to the smallest valid value to get correct BF values
    map_i = np.array(map_i)
    pix_ids = np.array(pix_ids)
    valid = np.isfinite(map_i)
    map_i[~valid] = np.amin(map_i[valid])

    # Keep track of the local best fit pixel before a possible rotation
    bf_pix = pix_ids[np.argmax(map_i)]
    bf_th, bf_phi = hp.pix2ang(NSIDE, bf_pix)
    map_info["bf_loc"] = {"azi": bf_phi, "zen": bf_th, "pix": bf_pix}
    print("  Current best fit local: " +
          u"azi={:.2f}°, zen={:.2f}°, pix={}".format(np.rad2deg(bf_phi),
                                                     np.rad2deg(bf_th),
                                                     bf_pix))

    # If we want equatorial coordinates, transform to new indices
    if coord == "equ":
        # If coord trafo previously failed, try again with all pixels now
        if trafo_next:
            # First we need to build a whole map if we use all pixels
            map_ = hp.ud_grade(map_in=logl_map, nside_out=NSIDE, power=0)
            map_[pix_ids] = map_i
            map_i = map_
            # Now we select all pixels from the full map to transform
            pix_ids = np.arange(hp.nside2npix(NSIDE))

        # Transform coordinates per pixel for the current map
        print("  Transform {} ".format(len(pix_ids)) +
              "pixels to equatorial coordinates.")
        try:
            trafo_next = False
            pix_ids = rotate_to_equ_pix(NSIDE, mjd, pix_ids)
        except RuntimeError:
            print("  Pixelization is too coarse. Try with all pixels in " +
                  "next iteration")
            trafo_next = True
            # But if we are already in the best resolution we have a problem
            if NSIDE == NSIDES[-1]:
                raise RuntimeError("Rotation is not bivariate, possibly due " +
                                   "to a too coarse pixelization. No higher " +
                                   "resolution seems available, so trafo " +
                                   "doesn't work for this scan.")
        else:
            # Also update accurate best fit in equatorial coordinates
            bf_ra, bf_dec = astro.dir_to_equa(zenith=[bf_th], azimuth=[bf_phi],
                                              mjd=mjd)

            bf_pix = rotate_to_equ_pix(NSIDE, mjd, bf_pix)[0]
            assert bf_pix == pix_ids[np.argmax(map_i)]
            map_info["bf_equ"] = {"ra": bf_ra[0], "dec": bf_dec[0],
                                  "pix": bf_pix}
            print("    Current best fit equ: " +
                  u"ra={:.2f}°, dec={:.2f}°, pix={}".format(
                      np.rad2deg(bf_ra)[0], np.rad2deg(bf_dec)[0], bf_pix))

    # Upscale and update map. Keep the vaues invariant (power=0)
    logl_map = hp.ud_grade(map_in=logl_map, nside_out=NSIDE, power=0)
    logl_map[pix_ids] = map_i

print("Combined map with NSIDES: [{}].".format(", ".join("{:d}".format(i)
                                                         for i in NSIDES)))
print("        Processed pixels: [{}].".format(
    ", ".join("{:d}".format(len(i)) for i in map_info["pixels"])))

# Check best fit sanity
if coord == "equ":
    _pix = np.argmax(logl_map)
    _th, _phi = hp.pix2ang(NSIDE, _pix)
    _dec = np.pi / 2. - _th
    assert _pix == map_info["bf_equ"]["pix"]
    # Rotation should not introduce errors > 2 * res ~ inside 2 pix
    assert np.isclose(_phi, map_info["bf_equ"]["ra"],
                      atol=2 * hp.nside2resol(NSIDE))
    assert np.isclose(_dec, map_info["bf_equ"]["dec"],
                      atol=2 * hp.nside2resol(NSIDE))
else:
    _pix = np.argmax(logl_map)
    _th, _phi = hp.pix2ang(NSIDE, _pix)
    assert _pix == map_info["bf_loc"]["pix"]
    assert _phi == map_info["bf_loc"]["azi"]
    assert _th == map_info["bf_loc"]["zen"]

# Smooth and normalize in normal LLH space if needed
if smooth > 0.:
    print(u"Smoothing the map with a {:.2f}° ".format(smooth) +
          "kernel and normalize as normal space PDF.")
    logl_map = smooth_and_norm(logl_map, np.deg2rad(smooth))

# Save in specified format with additional dict infos
# Plug in for default filename if no other was given
outf = outf.format(run_id, event_id)
if outfmt == "npy":
    print("Format is 'npy', so only the map array is saved.")
    fname = outf if outf.endswith(".npy") else outf + "." + outfmt
    np.save(fname, np.array(logl_map))
else:
    out_dict = {}
    out_dict.update(map_info)
    out_dict.update(ev_header)
    out_dict["map"] = list(logl_map)  # ndarray can't be serialized
    if outfmt == "json":
        # Takes significantly more space, but is human readable and portable
        fname = outf if outf.endswith(".json") else outf + "." + outfmt
        json.dump(out_dict, fp=open(fname, "w"), indent=1, sort_keys=True,
                  separators=(",", ":"))
    else:
        fname = outf if outf.endswith(".pickle") else outf + "." + outfmt
        pickle.dump(out_dict, open(fname, "w"))

print("Done. Saved map and info to:\n  {}".format(fname))

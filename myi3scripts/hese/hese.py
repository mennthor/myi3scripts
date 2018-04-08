# coding: utf8

from __future__ import print_function, division, absolute_import

import os
import re
import json
import gzip
import cPickle as pickle
from glob import glob
import numpy as np
import healpy as hp

from icecube import dataclasses, millipede, astro
from icecube.dataio import I3File

from ..healpy import theta_phi_to_dec_ra, cos_dist_equ
from ..healpy import healpy_map_loc_to_equa, smooth_and_norm_healpy_map
from ..misc import arr2str


def make_healpy_map_from_HESE_scan(infolder, scan_file_str, outfile,
                                   coord="equ", outfmt="json",
                                   smooth_sigma=None):
    """
    Takes a HESE scan folder and processes all scan files to build a healpy map
    and bundle map and scan information in ndarray or dict format. This builts
    maps in logl, so the best fit is the map maximum and thus assumes, that the
    scan logl values are also in log Likelihood and not negative log Likelihood.

    Find pass2 scan files at:
        /data/ana/Diffuse/HESE/Pass2_reconstruction/reconstruction_tracks

    Parameters
    ----------
    infolder : string
        Full path to the input scan folder. The scan folder is searched for I3
        files matching ``glob(scan_file_str)``.
    scan_file_str : string
        String with wildcard patterns used to identify the scan files in the
        ``infolder``. Originally this was ``SpiceMie_nside????.i3.bz2``.
    outfile : string
        Full filepath to save the output file to.
    coord : str, optional
        Which coordinate definition to use in the healpy map. Can be one of
        ``'equ' | 'local'``. Coordinates are mapped as follows:
        - If ``coord=='equ'``: ``dec=pi/2-theta`` and ``phi=ra``.
        - If ``coord=='local'``: ``zen=theta`` and ``azi=phi``.
    outfmt : str, optinal
        Can be one of ``'json' | 'pickle' | 'npy'``. If ``'json'`` or
        ``'pickle'``, then further information is saved in a dict with keys:
        - 'run_id' : Run ID of the event.
        - 'event_id' : Event ID of the event.
        - 'mjd' : MJD start tim of the event
        - 'NSIDES' : List of combined scanned resolutions.
        - 'pixels' : List of scanned pixels.
        - 'bf_loc' : Dictionary with keys:
            + 'azi' : Azimuth of best fit pixel, in radians.
            + 'zen' : Zenith of best fit pixel, in radians.
            + 'pix' : Pixel ID of the best fit for the highest included
              resolution.
        - 'bf_equ' : Only if '--coord' is `equ`. Dictionary with keys:
            + 'ra' : Right-ascension of best fit pixel, in radians.
            + 'dec' : Declination of best fit pixel, in radians.
            + 'pix' : Best fit pixel ID for the map transformed to equatorial
              coords.
        JSON files are compressed with gzip to save space. (default: 'json')
    smooth_sigma : float or None, optional
        Optional smoothing kernel width in degree. Originally, this was 1° for
        tracks and 30° for cascades to guess the influence of unknown
        systematics in the millipede scans. Gets passed to
        ``smooth_and_norm_healpy_map``. (default: ``None``)

    Returns
    -------
    out : ndarray or dict
        Depending on
    """
    # Check input arguments
    if coord not in ["equ", "local"]:
        raise ValueError("`coord` must be one of 'equ', 'local'.")
    if outfmt not in ["json", "npy", "pickle"]:
        raise ValueError("`outfmt` must be one of 'json', 'npy' or 'pickle'.")

    infolder = os.path.abspath(os.path.expandvars(os.path.expanduser(infolder)))
    outf = os.path.abspath(os.path.expandvars(os.path.expanduser(outfile)))

    # Make input file list, start at the lowest resolution
    srch_str = os.path.join(infolder, scan_file_str)
    f = sorted(glob(srch_str))
    fnames = map(os.path.basename, f)
    NSIDES = map(lambda s: int(re.search("nside([0-9]{4})", s).group(1)),
                 fnames)
    print("")
    print("Input folder is:\n  {}".format(infolder))
    print("Found scan files:\n  {}".format(arr2str(fnames)))
    print("  - Healpy resolutions: {}".format(arr2str(NSIDES), fmt="{:d}"))

    # Prepare scan and map info dicts
    ev_header = {"run_id": None, "event_id": None, "mjd": None}
    map_info = {"NSIDES": [], "pixels": []}

    # Init with the lowest healpy resolution possible and build maps from there
    logl_map = np.zeros(hp.nside2npix(1), dtype=float)

    # Loop through files and create the healpy arrays. For each resolution the
    # scanned pixels are injected into the previous upscaled map
    for i, fi in enumerate(f):
        NSIDE = NSIDES[i]
        NPIX = hp.nside2npix(NSIDE)
        print("Working on file: {}".format(fnames[i]))
        print("  - Resolution is: {}".format(NSIDE))

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

        # Get local map values
        map_vals, pix_ids = [], []
        while i3f.more():
            frame = i3f.pop_physics()
            assert frame["HealpixNSide"].value == NSIDE

            # Save millipede +lnLLH value in the temp map
            pix_id = frame["HealpixPixel"].value
            map_vals.append(-frame["MillipedeStarting2ndPassFitParams"].logl)
            pix_ids.append(pix_id)
        i3f.close()

        # If we are at the lowest resolution we should have scanned a full map
        if (i == 0) and not (len(pix_ids) == NPIX):
            raise ValueError("Lowest resolution scan doesn't contain all the " +
                             "pixels to build a full map. Check input files.")

        # Update verified map info
        map_info["NSIDES"].append(NSIDE)
        map_info["pixels"].append(pix_ids)

        # Set +-inf or nans to the smallest valid value to get correct best fit
        map_vals = np.array(map_vals)
        pix_ids = np.array(pix_ids)
        valid = np.isfinite(map_vals)
        map_vals[~valid] = np.amin(map_vals[valid])

        # Keep track of the local coordinate best fit pixel
        bf_pix = pix_ids[np.argmax(map_vals)]
        bf_th, bf_phi = hp.pix2ang(NSIDE, bf_pix)
        map_info["bf_loc"] = {"azi": bf_phi, "zen": bf_th, "pix": bf_pix}
        print("  Current best fit local: " +
              u"azi={:.2f}°, zen={:.2f}°, pix={}".format(
                  np.rad2deg(bf_phi), np.rad2deg(bf_th), bf_pix))

        # Upscale and update old coord map. Keep the values invariant (power=0)
        logl_map = hp.ud_grade(map_in=logl_map, nside_out=NSIDE, power=0)
        logl_map[pix_ids] = map_vals

    # If we want equatorial coordinates, transform map to new indices
    if coord == "equ":
        print("Transform complete local map to equatorial coordinates.")
        logl_map = healpy_map_loc_to_equa(logl_map, mjd)

        # Get exact transformation for the local best fit pixel
        bf_ra, bf_dec = astro.dir_to_equa(zenith=[bf_th], azimuth=[bf_phi], mjd=mjd)
        map_info["bf_equ"] = {"ra": bf_ra[0], "dec": bf_dec[0]}
        print("  Best fit equ    : ra={:.2f}°, dec={:.2f}°".format(
              np.rad2deg(bf_ra)[0], np.rad2deg(bf_dec)[0]))
        # Also store the best fit pixel in the new equatorial map, which might be
        # slightly off due to interpolation
        bf_pix = np.argmax(logl_map)
        bf_pix_ra, bf_pix_dec = theta_phi_to_dec_ra(*hp.pix2ang(NSIDE, bf_pix))
        map_info["bf_equ_pix"] = {"ra": bf_pix_ra[0], "dec": bf_pix_dec[0],
                                  "pix": bf_pix}
        print("  Best fit equ pix: ra={:.2f}°, dec={:.2f}°, pix={}".format(
              np.rad2deg(bf_pix_ra)[0], np.rad2deg(bf_pix_dec)[0], bf_pix))

    print("Combined map with NSIDES: [{}].".format(", ".join("{:d}".format(i)
                                                             for i in NSIDES)))
    print("        Processed pixels: [{}].".format(
        ", ".join("{:d}".format(len(i)) for i in map_info["pixels"])))

    # Check best fit sanity
    if coord == "equ":
        # Note: This might fail, when the smoothed map best fit strongly differs
        # from the unsmoothed logl best fit. This might happen, if the best fit
        # pixel is very isolated. Then it's value is smoothed down and a
        # different region may emerge as the best fit region, which is then also
        # the new max Likelihood region in the prior PDF map.

        # Get best fit values from map and compare to accurate best fit trafo
        bf_pix = np.argmax(logl_map)
        bf_th, bf_phi = hp.pix2ang(NSIDE, bf_pix)
        bf_dec, bf_ra = theta_phi_to_dec_ra(bf_th, bf_phi)
        # Get best accurate best fit to pixel best fit distance
        ra0, dec0 = map_info["bf_equ"]["ra"], map_info["bf_equ"]["dec"]
        cos_dist = cos_dist_equ(ra0, dec0, bf_ra, bf_dec)
        # cos_dist = (np.cos(bf_ra - ra0) * np.cos(bf_dec) * np.cos(dec0) +
        #             np.sin(bf_dec) * np.sin(dec0))
        # cos_dist = np.clip(cos_dist, -1., 1.)
        dist = np.arccos(cos_dist)

        # Get max neighbour distance to get the allowed deviation
        _th, _phi = hp.pix2ang(NSIDES[-1], hp.get_all_neighbours(
            nside=NSIDES[-1], theta=bf_th, phi=bf_phi))
        _dec, _ra = theta_phi_to_dec_ra(_th, _phi)
        cos_dist = cos_dist_equ(_ra, _dec, bf_ra, bf_dec)
        # cos_dist = np.clip((np.cos(_ra - bf_ra) * np.cos(_dec) * np.cos(bf_dec) +
        #                      np.sin(_dec) * np.sin(bf_dec)), -1., 1.)
        max_dist = np.amax(np.arccos(cos_dist))
        # Rotation should not have introduced angular errors greater than the
        # maximum distance to all neighbour pixels
        assert dist <= max_dist
        # Best fit from single maps should also be the global best fit
        assert bf_pix == map_info["bf_equ"]["pix"]
    else:
        bf_pix = np.argmax(logl_map)
        bf_th, bf_phi = hp.pix2ang(NSIDE, bf_pix)
        # Best fit from single maps should also be the global best fit
        assert bf_pix == map_info["bf_loc"]["pix"]
        assert bf_phi == map_info["bf_loc"]["azi"]
        assert bf_th == map_info["bf_loc"]["zen"]

    # Smooth and normalize in normal LLH space if needed
    if smooth_sigma > 0.:
        print(u"Smoothing the map with a {:.2f}° ".format(smooth_sigma) +
              "kernel and normalize as normal space PDF.")
        logl_map = smooth_and_norm_healpy_map(logl_map,
                                              np.deg2rad(smooth_sigma))

    # Save in specified format with additional dict infos
    if outfmt == "npy":
        print("Format is 'npy', so only the map array is saved.")
        fname = outf if outf.endswith(".npy") else outf + "." + outfmt
        np.save(fname, np.array(logl_map))
    else:
        out_dict = {}
        out_dict.update(map_info)
        out_dict.update(ev_header)
        if outfmt == "json":
            # Takes significantly more space, but is human readable and portable
            out_dict["map"] = list(logl_map)  # ndarray can't be JSON serialized
            fname = outf if outf.endswith(".json") else outf + "." + outfmt
            with gzip.open(fname, "wb") as _outfile:
                json.dump(out_dict, fp=_outfile, indent=1, sort_keys=True,
                          separators=(",", ":"))
        else:
            out_dict["map"] = logl_map
            fname = outf if outf.endswith(".pickle") else outf + "." + outfmt
            with open(fname) as _outfile:
                pickle.dump(out_dict, _outfile)

    print("Done. Saved map to:\n  {}".format(fname))

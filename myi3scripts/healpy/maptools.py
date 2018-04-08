# coding: utf8

from __future__ import division, absolute_import

import numpy as np
import healpy as hp

from icecube import astro

from .coords import theta_phi_to_dec_ra


def healpy_map_loc_to_equa(hp_map, mjd):
    """
    Transforms a healpy map from local coordinates, where ``azimuth=phi`` and
    ``zenith=theta``, into a equatorial coordinate map with convention
    ``dec=pi/2-theta`` and ``phi=ra``.

    We can then extract equatorial coordinates from the new map by a simple
    transformation ``dec=pi/2-theta``. This saves time durimg evaluation but
    introduces an error, beause interpolation is needed during the conversion.

    Note: We have to do the process backwards so we have to interpolate values,
    because when rotating in declination, we don't map pixels in a bivariate
    fashion as the number of pixels in a dec band varies with declination.

    Parameters
    ----------
    hp_map : array-like
        Healpy map in local coordinates.
    mjd : float
        MJD event time needed to transform local to equa. coordinates.

    Returns
    -------
    hp_map : array-like
        Rotated map so that ``dec=pi/2-theta`` and ``ra=phi``. s
    """
    NSIDE = hp.get_nside(hp_map)
    NPIX = hp.nside2npix(NSIDE)

    # Make accurate ra, dec coordinates from selected pixels
    pix = np.arange(NPIX)
    th, phi = hp.pix2ang(NSIDE, pix)
    dec, ra = theta_phi_to_dec_ra(th, phi)

    # Transform to local coordinates and get interpolated map values to
    # avoid non-bivariat mapping of discrete pixels
    zen, azi = astro.equa_to_dir(ra=ra, dec=dec, mjd=mjd)
    hp_map = hp.get_interp_val(hp_map, theta=zen, phi=azi)
    return hp_map


def smooth_and_norm_healpy_map(logl_map, smooth_sigma=None):
    """
    Takes a lnLLH map, converts it to normal space, applies gaussian smoothing
    and normalizes it, so that the integral over the unit sphere is 1.

    Parameters
    ----------
    logl_map : array-like
    ealpy map array.
    smooth_sigma : float or None, optional
        Width in sigma of gaussian smoothing kernel, must be ``>0.``.
        (default: None)

    Returns
    -------
    pdf_map : array-like
        Smoothed and normalized spatial PDF map.
    """
    if smooth_sigma < 0.:
        raise ValueError("`smooth_sigma` can be in range [0, *].")

    # Normalize to sane values in [*, 0] for conversion llh = exp(logllh)
    pdf_map = np.exp(logl_map - np.amax(logl_map))

    # Smooth with a gaussian kernel
    pdf_map = hp.smoothing(map_in=pdf_map, sigma=smooth_sigma, verbose=False)
    # Healpy smoothing may produce numerical erros, so fix them after smoothing
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

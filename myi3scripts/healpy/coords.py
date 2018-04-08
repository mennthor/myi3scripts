# coding: utf8

from __future__ import division, absolute_import

import numpy as np


def dec_ra_to_theta_phi(dec, ra):
    """
    Convert equatorial coordinates ``dec, ra`` to healpy coordinates
    ``theta, phi`` using the convention ``dec=pi/2-theta`` and ``phi=ra``.
    """
    return np.pi / 2. - dec, ra


def theta_phi_to_dec_ra(th, phi):
    """
    Convert healpy coordinates ``theta, phi`` to equatorial coordinates
    ``dec, ra`` using the convention ``theta=pi/2-dec`` and ``ra=phi``.
    """
    return np.pi / 2. - th, phi


def cos_dist_equ(ra0, dec0, ra1, dec1):
    """
    Cosine of great circle distance in equatorial coordinates.
    Values get clipped at ``[-1, 1]`` to repair float errors at the edges.
    """
    cos_dist = (np.cos(ra1 - ra0) * np.cos(dec1) * np.cos(dec0) +
                np.sin(dec1) * np.sin(dec0))
    return np.clip(cos_dist, -1., 1.)

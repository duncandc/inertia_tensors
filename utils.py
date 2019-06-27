r"""
A set of utilty functions
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np

__all__=['sample_ellipsoidal_surface',
         'sample_ellipsoidal_volume']
__author__ = ['Duncan Campbell']


def sample_ellipsoidal_surface(npts, semi_axes):
    r"""
    return points sampled uniformly on the surface of a hyperellipsoid

    Parameters
    ----------
    npts : int
        number of points to sample

    semi_axes : array_like
        arry of shape (ndim,) of principle semi-axis lengths

    Returns
    -------
    result : numpy.ndarray
        array of shape (npts, ndim) of points
    """

    semi_axes = np.sort(np.atleast_1d(semi_axes))
    ndim = len(semi_axes)

    if ndim <= 1:
        raise ValueError('hyperellipsoid dimension must be >=2.')

    x = np.random.normal(size=(npts,ndim), scale=semi_axes)
    r = np.sqrt(np.sum((x/semi_axes)**2, axis=-1))

    return (1.0/r[:,np.newaxis])*x


def sample_ellipsoidal_volume(npts, semi_axes):
    r"""
    return points sampled uniformly within a hyperellipsoidal volume

    Parameters
    ----------
    npts : int
        number of points to sample

    semi_axes : array_like
        arry of shape (ndim,) of principle semi-axis lengths

    Returns
    -------
    result : numpy.ndarray
        array of shape (npts, ndim) of points
    """

    semi_axes = np.sort(np.atleast_1d(semi_axes))[::-1]
    ndim = len(semi_axes)

    if ndim <= 1:
        raise ValueError('hyperellipsoid dimension must be >=2.')

    x = np.random.normal(size=(npts,ndim), scale=semi_axes)
    r = np.sqrt(np.sum((x/semi_axes)**2, axis=-1))

    ran_scale = (np.random.random(npts))**(1.0/ndim)

    r = r/ran_scale
    return (1.0/r[:,np.newaxis])*x




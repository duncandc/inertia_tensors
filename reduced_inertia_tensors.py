"""
functions to calculate sets of inertia tensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


__all__ = ('reduced_inertia_tensors')
__author__ = ('Duncan Campbell')


def reduced_inertia_tensors(x, weights=None):
    r"""
    Calculate the n1 inertia tensors for a set of n2 points of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1,ndim,ndim) of inertia tensors

    Examples
    --------
    """

    n1, n2, ndim = np.shape(x)


    if weights is None:
        weights = np.ones((n1,n2))
    elif np.shape(weights) != (n1,n2):
        msg = ('weights array must be of shape (n1,n2)')
        raise ValueError(msg)

    r_squared = np.sum(x**2, -1)
    I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights[:,:,np.newaxis])
    return I/(np.ones((n1,ndim,ndim))*n2)

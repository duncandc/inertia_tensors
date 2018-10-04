"""
functions to calculate sets of inertia tensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


__all__ = ('inertia_tensors')
__author__ = ('Duncan Campbell')


def inertia_tensors(x, weights=None):
    r"""
    Calculate the n1 inertia tensors for a set of n2 points of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.  If an array of shape (n2, ndim) points is passed
        n1 is assumed to be equal to 1.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1,ndim,ndim) of inertia tensors

    Examples
    --------
    """

    if len(np.shape(x))==2:
        x = x[np.newaxis,:,:]

    n1, n2, ndim = np.shape(x)

    if weights is None:
        weights = np.ones((n1,n2))
    elif np.shape(weights) == (n2,):
        weights = weights[np.newaxis,:]

    if np.shape(weights) != (n1,n2):
        msg = ('weights array must be of shape (n1,n2)')
        raise ValueError(msg)

    # copy the weights ndim times along a new axis
    # in order to make them the same shape as x
    weights = np.repeat(weights[:,:, np.newaxis], ndim, axis=2)

    I = np.einsum('...ij,...ik->...jk', x, x*weights)
    return I/(np.ones((n1,ndim,ndim))*n2)

"""
function to calculate sets of inertia tensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from rotations.rotations2d import rotation_matrices_from_basis as rotation_matrices_from_basis_2d
from rotations.rotations2d import rotate_vector_collection as rotate_vector_collection_2d
from rotations.rotations3d import rotation_matrices_from_basis as rotation_matrices_from_basis_3d
from rotations.rotations2d import rotate_vector_collection as rotate_vector_collection_3d


__all__ = ('inertia_tensors',
           'reduced_inertia_tensors',
           'iterative_inertia_tensors')
__author__ = ('Duncan Campbell')


def inertia_tensors(x, weights=None):
    r"""
    Calculate the inertia tensors for n1 sets, of n2 points, of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.  If an array of shape (n2, ndim) points is passed,
        n1 is assumed to be equal to 1.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights.
        Default sets weights argument to np.ones((n1,n2)).

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1, ndim, ndim) storing the n1 inertia tensors

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


def reduced_inertia_tensors(x, weights=None):
    r"""
    Calculate reduced inertia tensors for n1 sets of n2 points of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.  If an array of shape (n2, ndim) points is passed,
        n1 is assumed to be equal to 1.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights.
        Default sets weights argument to np.ones((n1,n2)).

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1, ndim, ndim) storing the n1 inertia tensors

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

    r_squared = np.sum(x**2, -1)
    I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
    return I/(np.ones((n1,ndim,ndim))*n2)


def iterative_inertia_tensors(x, weights=None, rtol=0.01, niter_max=100):
    r"""
    Calculate iterative reduced inertia tensors for n1 sets of n2 points of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.  If an array of shape (n2, ndim) points is passed,
        n1 is assumed to be equal to 1.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights.
        Default sets weights argument to np.ones((n1,n2)).

    rtol : float
        relative tolerance on eignevalues of the inertia tensors

    niter_max : int
        maximum nmumber of iterations to perform

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1, ndim, ndim) storing the n1 inertia tensors

    Examples
    --------
    """

    if len(np.shape(x))==2:
        x = x[np.newaxis,:,:]

    n1, n2, ndim = np.shape(x)

    if ndim == 2:
        rot_func = rotation_matrices_from_basis_2d
        rotate_vector_collection = rotate_vector_collection_2d
    elif ndim == 3:
        rot_func = rotation_matrices_from_basis_3d
        rotate_vector_collection = rotate_vector_collection_3d
    else:
        msg = ('the iterative reduced inertia tensor only works with ndim = 2 or 3.')
        raise ValueError(msg)

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

    r_squared = np.sum(x**2, -1)
    I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
    I = I/(np.ones((n1,ndim,ndim))*n2)

    while niter < niter_max:
        evals, evecs = np.linalg.eigh(I)
        evecs = evecs[::-1] # put in decending order
        evals = evals[::-1]

        rot = rot_func(*evecs)
        xx = rotate_vector_collection(rot, x)
        r_squared = np.sum((xx/evals)**2, -1)

        I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
        I = I/(np.ones((n1,ndim,ndim))*n2)

    return I












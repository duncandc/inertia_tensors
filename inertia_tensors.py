"""
function to calculate sets of inertia tensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from rotations.rotate_vector_collection import rotate_vector_collection
from rotations.rotations2d import rotation_matrices_from_basis as rotation_matrices_from_basis_2d
from rotations.rotations3d import rotation_matrices_from_basis as rotation_matrices_from_basis_3d


__all__ = ('inertia_tensors',
           'reduced_inertia_tensors',
           'iterative_inertia_tensors')
__author__ = ('Duncan Campbell')


def _process_args(x, weights):
    """
    process arguments for inertia tensor functions
    """

    if len(np.shape(x))==2:
        x = x[np.newaxis,:,:]

    x = np.atleast_1d(x)

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

    return x, weights


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
    x, weights = _process_args(x, weights)
    n1, n2, ndim = np.shape(x)

    I = np.einsum('...ij,...ik->...jk', x, x*weights)
    m = np.sum(weights, axis=1)
    return I/(np.ones((n1,ndim,ndim))*m[:,np.newaxis])


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

    x, weights = _process_args(x, weights)
    n1, n2, ndim = np.shape(x)

    r_squared = np.sum(x**2, -1)
    
    # ignore points at r=0
    mask = (r_squared==0.0)
    weights[mask] = 0.0
    r_squared[mask] = 1.0
    
    I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
    m = np.sum(weights, axis=1)
    return I/(np.ones((n1,ndim,ndim))*m[:,np.newaxis])


def iterative_inertia_tensors(x, weights=None, rtol=0.01, niter_max=5):
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
        relative tolerance on axis ratios
        the calculate will continue while any axis ratio changes by more than rtol
        between iterations.

    niter_max : int
        maximum nmumber of iterations to perform

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1, ndim, ndim) storing the n1 inertia tensors

    Examples
    --------
    """

    x, weights = _process_args(x, weights)
    n1, n2, ndim = np.shape(x)

    if ndim == 2:
        rot_func = rotation_matrices_from_basis_2d
    elif ndim == 3:
        rot_func = rotation_matrices_from_basis_3d
    else:
        msg = ('the iterative reduced inertia tensor only works with ndim = 2 or 3.')
        raise ValueError(msg)

    # calculate intial inertia tensor
    r_squared = np.sum(x**2, -1)

    # ignore points at r=0
    mask = (r_squared==0.0)
    weights[mask] = 0.0
    r_squared[mask] = 1.0

    I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
    m = np.sum(weights, axis=1)
    I = I/(np.ones((n1,ndim,ndim))*m[:,np.newaxis])
    
    evals, evecs = np.linalg.eigh(I)
    # put in order a,b,c
    evecs = evecs[:,::-1,:]
    evals = np.sqrt(evals[:,::-1])

    # ellipsoidal volume
    v0 = 4.0/3.0*np.pi*np.prod(evals,axis=-1)

    # intial axis ratios, a/a, b/a, c/a
    axis_ratios0 = evals/evals[:,0,np.newaxis]

    niter = 1  # iteratively calculate I
    exit=False
    while (niter < niter_max) & (exit==False):

        # calculate rotation matrix between eigen basis and axis-aligned basis
        evecs = [evecs[:,i,:] for i in range(ndim)]  # re-arrange eigenvalues
        rot = rot_func(*evecs)
        rot = np.linalg.inv(rot)

        # rotate distribution to align with axis
        xx = rotate_vector_collection(rot, x)

        # calculate elliptical radial distances
        r_squared = np.sum((xx/evals[:,np.newaxis])**2, -1)
    
        # ignore points at r=0
        mask = (r_squared==0.0)
        weights[mask] = 0.0
        r_squared[mask] = 1.0

        I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights)
        m = np.sum(weights, axis=1)
        I = I/(np.ones((n1,ndim,ndim))*m[:,np.newaxis])

        # calculate eignenvectors and values
        # note that eigh() returns minor axis values first
        evals, evecs = np.linalg.eigh(I)
        # put in order a,b,c
        evecs = evecs[:,::-1,:]
        evals = np.sqrt(evals[:,::-1])

        # re-scale axis to maintain constant volume
        v = 4.0/3.0*np.pi*np.prod(evals,axis=-1)
        scale = v0/v
        evals = evals*scale[:,np.newaxis]

        # calculate axis ratios
        axis_ratios = evals/evals[:,0,np.newaxis]
        da = np.fabs(axis_ratios - axis_ratios0)/axis_ratios0
        if np.max(da)<=rtol:
            exit = True

        print(da, evals)

        axis_ratios0 = axis_ratios
        niter += 1

    return I












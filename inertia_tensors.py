"""
function to calculate sets of inertia tensors
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from rotations.rotate_vector_collection import rotate_vector_collection
from rotations.rotations2d import rotation_matrices_from_basis as rotation_matrices_from_basis_2d
from rotations.rotations3d import rotation_matrices_from_basis as rotation_matrices_from_basis_3d
from rotations.vector_utilities import angles_between_list_of_vectors


__all__ = ('inertia_tensors',
           'reduced_inertia_tensors',
           'iterative_inertia_tensors_3D')
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

    if np.shape(weights) != (n1,n2,ndim):
        # copy the weights ndim times along a new axis
        # in order to make them the same shape as x
        weights = np.repeat(weights[:,:, np.newaxis], ndim, axis=2)

    return x, weights


def _principal_axes_3D(I):
    """
    Return the principle axes and half-lengths of an ellipsoid defined by I

    Returns
    -------
    A, B, C : numpy.arrays
        arrays of the primary, intermediate, and minor axis lengths

    Av, Bv, Cv : numpy.arrays
        arrays of primary, intermediate, and minor eigenvectors
    """

    # note that eigh() returns the axes in ascending order
    evals, evecs = np.linalg.eigh(I)

    evecs = evecs[:,:,::-1]

    Av = evecs[:,:,0]
    Bv = evecs[:,:,1]
    Cv = evecs[:,:,2]

    evals = np.sqrt(evals[:,::-1])

    A = evals[:,0]
    B = evals[:,1]
    C = evals[:,2]

    return A, B, C, Av, Bv, Cv




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


def iterative_inertia_tensors_3D(x, weights=None, rtol=0.01, niter_max=5):
    r"""
    Calculate iterative reduced inertia tensors for n1 sets of n2 points of dimension 3.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, 3) storing n1 sets of n2 points
        of dimension ndim.  If an array of shape (n2, 3) points is passed,
        n1 is assumed to be equal to 1.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights.
        Default sets weights argument to np.ones((n1,n2)).

    rtol : float
        Relative tolerance on axis ratios. The calculation will continue
        while any axis ratio fractiolnally changes between two iterations by more than rtol

    niter_max : int
        maximum nmumber of iterations to perform

    Returns
    -------
    I : numpy.ndarray
        an array of shape (n1, 3, 3) storing the n1 inertia tensors

    Examples
    --------
    """

    x, weights = _process_args(x, weights)
    n1, n2, ndim = np.shape(x)

    rot_func = rotation_matrices_from_basis_3d

    I = reduced_inertia_tensors(x, weights)
    A, B, C, Av, Bv, Cv = _principal_axes_3D(I)

    # intial ellipsoidal volume
    ellipsoid_volume_0 = (4.0/3.0)*np.pi*A*B*C

    # intial axis ratios
    b_to_a_0, c_to_a_0 = B/A, C/A
    Av_0 = Av

    niter = 1  # iteratively calculate I
    exit=False
    while (niter < niter_max) & (exit==False):

        # calculate rotation matrix between eigen basis and axis-aligned basis
        rot = rot_func(Av, Bv, Cv)
        inv_rot = np.linalg.inv(rot)

        # rotate distribution to align with axis
        xx = rotate_vector_collection(inv_rot, x)

        # calculate ellipsoidal radial distances
        axis_ratios = np.vstack((A,B,C)).T
        norm = np.repeat(axis_ratios[:,np.newaxis,:], n2, axis=1)
        r_squared = np.sum((xx/norm)**2, -1)

        # ignore points at r=0
        mask = (r_squared==0.0)
        weights[mask] = 0.0
        r_squared[mask] = 1.0

        # calculate eigen tensors
        I = np.einsum('...ij,...ik->...jk', xx/(r_squared[:,:,np.newaxis]), xx*weights)
        m = np.sum(weights, axis=1)
        I = I/(np.ones((n1,ndim,ndim))*m[:,np.newaxis])

        A, B, C, Av, Bv, Cv = _principal_axes_3D(I)

        # rotate back into original frame
        Av = rotate_vector_collection(rot, Av)
        Bv = rotate_vector_collection(rot, Bv)
        Cv = rotate_vector_collection(rot, Cv)

        # re-scale axes to maintain constant volume
        ellipsoid_volume = (4.0/3.0)*np.pi*A*B*C
        f = (1.0*ellipsoid_volume/ellipsoid_volume_0)
        A = A*f**(-1.0/3.0)
        B = B*f**(-1.0/3.0)
        C = C*f**(-1.0/3.0)

        # calculate axis ratios
        b_to_a, c_to_a = B/A, C/A
        da_1 = np.fabs(b_to_a - b_to_a_0)/b_to_a_0
        da_2 = np.fabs(c_to_a - c_to_a_0)/c_to_a_0
        if (np.max(da_1)<=rtol) & (np.max(da_2)<=rtol):
            exit = True

        # angle between primary eigenvectors
        theta = np.degrees(angles_between_list_of_vectors(Av, Av_0))

        # update parameters
        b_to_a_0 = b_to_a
        c_to_a_0 = c_to_a
        Av_0 = Av
        niter += 1

    # re-construct inertia tensor
    m = np.tile(np.identity(3), (n1,1,1))
    m[:,0,0] = A**2
    m[:,1,1] = B**2
    m[:,2,2] = C**2

    s = np.zeros((n1,3,3))
    s[:,:,0] = Av
    s[:,:,1] = Bv
    s[:,:,2] = Cv

    I = np.matmul(np.matmul(s,m),s.transpose(0,2,1))

    # check reconstruction
    evals, evecs = np.linalg.eigh(I)
    assert np.allclose(np.sqrt(evals[:,0]),C)
    assert np.allclose(np.sqrt(evals[:,1]),B)
    assert np.allclose(np.sqrt(evals[:,2]),A)

    return I












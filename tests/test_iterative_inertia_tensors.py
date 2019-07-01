"""
"""
import numpy as np
from inertia_tensors.inertia_tensors import iterative_inertia_tensors_3D
from inertia_tensors.utils import sample_ellipsoidal_volume


__all__ = ('test_1',)


def test_1():
    """
    basic 3-d unit test
    """

    n1 = 10
    n2 = 100
    ndim = 3

    semi_axes = np.random.random((n1, ndim))
    coords = np.array([sample_ellipsoidal_volume(n2, semi_axes[i]) for i in range(0, n1)])

    Is = iterative_inertia_tensors_3D(coords)

    assert np.shape(Is) == (n1, ndim, ndim)


def test_2():
    """
    test weights
    """

    n1 = 10
    n2 = 100
    ndim = 3

    semi_axes = np.random.random((n1, ndim))
    coords = np.array([sample_ellipsoidal_volume(n2, semi_axes[i]) for i in range(0, n1)])

    weights = np.ones((n1, n2))

    Is_1 = iterative_inertia_tensors_3D(coords, weights=weights)
    Is_2 = iterative_inertia_tensors_3D(coords, weights=None)

    assert np.shape(Is_1) == (n1, ndim, ndim)
    assert np.shape(Is_2) == (n1, ndim, ndim)
    assert np.allclose(Is_1, Is_2)


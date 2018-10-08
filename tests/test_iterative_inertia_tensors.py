"""
"""
import numpy as np
from ..inertia_tensors import iterative_inertia_tensors
from ..utils import sample_ellipsoidal_volume


__all__ = ('test_1',)


def test_1():
    """
    basic 2-d unit test
    """

    n1 = 10
    n2 = 100
    ndim = 2

    semi_axes = np.random.random((n1,ndim))
    coords = np.array([sample_ellipsoidal_volume(n2, semi_axes[i]) for i in range(0,n1)])

    Is = iterative_inertia_tensors(coords)

    assert np.shape(Is)==(n1,ndim,ndim)


def test_2():
    """
    basic 3-d unit test
    """

    n1 = 10
    n2 = 100
    ndim = 3

    semi_axes = np.random.random((n1,ndim))
    coords = np.array([sample_ellipsoidal_volume(n2, semi_axes[i]) for i in range(0,n1)])

    Is = iterative_inertia_tensors(coords, )

    assert np.shape(Is)==(n1,ndim,ndim)

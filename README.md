# Inertia Tensors

This package contains functions to calculate [inertia tensors](https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor) for collections of n-dimensional points.
![](./notebooks/ellipses_plot.png)


## Description

This package contains functions that calculate the:

* standard inertia tensor
* reduced inertia tensor
* iterative reduced inertia tensor


## Requirements

In order to use the functions in this package, you will need the following Python packages installed:

* [numpy](http://www.numpy.org)
* [astropy](http://www.astropy.org)
* [rotations](https://github.com/duncandc/rotations)


## Installation

Place this directory in your PYTHONPATH.  The various functions can then be imported as, e.g.:

```
from inertia_tensors import inertia_tensors
```

You can run the testing suite for this package using [pytest](https://docs.pytest.org/en/latest/) framwork by executing the following command in the package directory:

```
pytest
```


contact:
duncanc@andrew.cmu.edu

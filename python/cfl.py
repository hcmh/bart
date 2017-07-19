# Copyright 2013-2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
# 2017 Christian Holme <christian.holme@med.uni-goettingen.de>


import numpy as np

def readcfl(name):
    """
    Read cfl-file into numpy-array.

    Parameters
    ----------
    name: str
        name of the cfl-file, without '.cfl' or '.hdr'

    Returns
    -------
    numpy.ndarray
        The array in complex64 format
    """

    # get dims from .hdr
    with open(name + ".hdr", "r") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    dims_prod = np.cumprod(dims)
    n = dims_prod[-1]
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "r") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major

	
def writecfl(name, array):
    """
    Write numpy array to cfl-file.

    Parameters
    ----------
    name: str
        name of the cfl-file, without '.cfl' or '.hdr'
    array: array_like
        The numpy array to be written to disk

    Returns
    -------
    None
    """
    with open(name + ".hdr", "w") as h:
        h.write('# Dimensions\n')
        for i in (array.shape):
                h.write("%d " % i)
        h.write('\n')
    with open(name + ".cfl", "w") as d:
        array.T.astype(np.complex64).tofile(d) # tranpose for column-major order

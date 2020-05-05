# Copyright 2013-2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>


import numpy as np
import mmap
import os

def readcfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major


def writecfl(name, array):
    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        for i in (array.shape):
                h.write("%d " % i)
        h.write('\n')

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.truncate(d.fileno(), size)
        with mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE) as mm:
            mm.write(array.astype(np.complex64).tobytes(order='F'))

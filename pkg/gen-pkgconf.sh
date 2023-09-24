#!/bin/bash


# Thanks to:
# https://people.freedesktop.org/~dbn/pkg-config-guide.html#writing

# Example:

# prefix=/usr/local
# exec_prefix=${prefix}
# includedir=${prefix}/include
# libdir=${exec_prefix}/lib
# 
# Name: foo
# Description: The foo library
# Version: 1.0.0
# Cflags: -I${includedir}/foo
# Libs: -L${libdir} -lfoo

set -euo pipefail

BART_PREFIX="$1"
shift
CFLAGS="$1"
shift
LIBS="$1"
shift
VER="$1"
shift
OUT="$1"
shift


cat << EOF > "$OUT"

prefix=$BART_PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include/bart
libdir=\${prefix}/lib/bart
commanddir=\${libdir}/commands

Name: bart
Description: Toolbox for computational magnetic resonance imaging
Version: $VER
Cflags: $CFLAGS
Libs: $LIBS
EOF




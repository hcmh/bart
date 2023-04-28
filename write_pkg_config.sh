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

TOOLBOX_PATH="$1"
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
TOOLBOX_PATH=$TOOLBOX_PATH
exec_prefix=${TOOLBOX_PATH}
includedir=${TOOLBOX_PATH}/src
libdir=${TOOLBOX_PATH}/lib

Name: bart
Description: Toolbox for computational magnetic resonance imaging
Version: $VER
Cflags: $CFLAGS
Libs: $LIBS
EOF




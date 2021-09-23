#!/bin/bash
# Copyright 2021. Martin Uecker.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2021 Moritz Blumenthal <moritz.blumenthal@med.uni-goettingen.de>
#
# Code to runn bart commands in loop
#
set -e

LOGFILE=/dev/stdout

helpstr=$(cat <<- EOF
-p paralellize loop
-i input files
-o output files
-h help
EOF
)

usage="Usage: $0 [-hpi:o:] loop_dim \"bart command\""

ifiles=
ofiles=

para="FALSE"

while getopts "hpi:o:" opt; do
        case $opt in
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	i)
		ifiles+=" $OPTARG"
	;;
	o)
		ofiles+=" $OPTARG"
	;;
	p)
		para="TRUE"
	;;
        \?)
        	echo "$usage" >&2
		exit 1
        ;;
        esac
done

shift $((OPTIND - 1))


if [ $# -lt 2 ] ; then

        echo "$usage" >&2
        exit 1
fi

#WORKDIR=$(mktemp -d)
# Mac: http://unix.stackexchange.com/questions/30091/fix-or-alternative-for-mktemp-in-os-x
WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT

ldim=$1
comd=$2

lsize=0
j=0
cmd2=$comd


for file in $ifiles
do
	if [ 0 -eq $lsize ]
	then
		lsize=$(bart show -d$ldim $file)
	fi

	if ! [ $(bart show -d$ldim $file) -eq $lsize ]
	then
		echo "Loop dimensions are not equal!"
		exit 1
	fi

	if [ "$(echo "$cmd2" | sed -e "s|$file|$WORKDIR/infile_${j}_IDX|")" == "$cmd2" ]
	then
		echo "No replacement for input is found!"
		exit 1
	fi

	cmd2="$(echo "$cmd2" | sed -e "s|$file|$WORKDIR/infile_${j}_IDX|")"

	for i in $(seq $lsize)
	do
		bart slice $ldim $((i-1)) $file $WORKDIR/infile_${j}_${i}
	done
	j=$((j+1))
done


j=0
for file in $ofiles
do
	if [ "$(echo "$cmd2" | sed -e "s|$file|$WORKDIR/outfile_${j}_IDX|")" == "$cmd2" ]
	then
		echo "No replacement for input is found!"
		exit 1
	fi

	cmd2="$(echo "$cmd2" | sed -e "s|$file|$WORKDIR/outfile_${j}_IDX|")"

	j=$((j+1))
done

if [ $para == TRUE ]
then
	for i in $(seq $lsize)
	do
		${cmd2//IDX/$i} &
	done
	wait
else
	for i in $(seq $lsize)
	do
		${cmd2//IDX/$i}
	done
fi

j=0
for file in $ofiles
do
	join=

	for i in $(seq $lsize)
	do
		join+=" $WORKDIR/outfile_${j}_${i}"
	done

	bart join $ldim $join $file

	j=$((j+1))
done

exit 0

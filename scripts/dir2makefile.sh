#!/bin/bash
set -euo pipefail
set -B

title=$(cat <<- EOF
	dir2makefile v0.1 (of the Berkeley Advanced Reconstruction Toolbox)
EOF
)

helpstr=$(cat <<- EOF
This script takes a directory of bart .cfl and .hdr files,
extracts the commands used to create them, and creates a Makefile
which reproduces them.
If there is second argument, the output will be written to that file.
Otherwise, a file called Makefile is created in the input directory.
The entire directory can then be reproced by running
\tmake -B *.hdr
or, in parallel, with
\tmake -Bj *.hdr
(here, -B instructs make to unconditionally recreate targets).
A script can be created using
\tmake -Bn *.hdr | grep -v "make:"

-h help
EOF
)

usage="Usage: $(basename $0)) [-h] <INDIR> [Makefile_output]"

echo "$title"

while getopts "h" opt; do
        case $opt in
	h)
		echo "$usage"
		echo
		echo -e "$helpstr"
		exit 0
	;;
        \?)
		echo "$usage" >&2
		exit 1
        ;;
        esac
done

shift $((OPTIND - 1))


if [[ $# -lt 1 || $# -gt 2 ]] ; then

        echo "$usage" >&2
        exit 1
fi



INDIR="$1"

if [ $# -ge 2 ]; then
	OUTF="$2"
else
	OUTF="${INDIR}/Makefile"
fi


DEBUG=false
if [[ -v DEBUG_LEVEL ]] ; then
	if [[ "$DEBUG_LEVEL" -ge 5 ]] ; then
		DEBUG=true
	fi
fi

truncate -s0 $OUTF


declare -A IOs
declare -A CMDLs
declare -A VERs

for hdr in ${INDIR}/*.hdr ; do

	CMDLs["$hdr"]=$(sed '4q;d' "$hdr")
	IOs["$hdr"]=$(sed '6q;d' "$hdr")
	VERs["$hdr"]=$(sed '8q;d' "$hdr")
done


#check uniqueness of command lines

declare -A IOfs
declare -A CMDLfs

for hdr in ${!CMDLs[@]}; do
	# check if current comandline is already in filtered array
	$DEBUG && echo $hdr
	dup=false
	hdrdup=''
	for hdrtest in "${!CMDLfs[@]}" ; do
		cmdl="${CMDLfs[$hdrtest]}"
		if [[ "$cmdl" == "${CMDLs[$hdr]}" ]] ; then
			dup=true
			hdrdup="$hdrtest"
			break
		fi
	done
	if $dup ; then
		$DEBUG && echo "Duplicate found! $hdr $hdrtest"

		# check which has more outputs, based on stringlength
		hIO="${IOs[$hdr]}"
		htIO="${IOs[$hdrtest]}"

		lh=${#hIO}
		lht=${#htIO}
		if [ $lh -gt $lht ] ; then
			# the new hdr has more outputs
			# so we replace it in filtered associative array
			unset CMDLfs[$hdrtest]
			unset IOfs[$hdrtest]
			IOfs["$hdr"]="${IOs[$hdr]}"
			CMDLfs["$hdr"]="${CMDLs[$hdr]}"
		else
			# the new hdr has fewer outputs
			# so we just ignore it
			true
		fi


	else
		IOfs["$hdr"]="${IOs[$hdr]}"
		CMDLfs["$hdr"]="${CMDLs[$hdr]}"
	fi

done

printf "all:\n\n" >> $OUTF

printf "BART?=bart\n">>$OUTF
# get a version from the first header
# Whatever that means for an associative array
VERsU=("${VERs[@]}")
printf "#Originally created using: ${VERsU[0]}\n\n" >> $OUTF

for hdr in ${!CMDLfs[@]}; do

	IO=${IOfs["$hdr"]}
	cmdl=${CMDLfs["$hdr"]}

	# get inputs and output of cmdline	
	INs=()
	OUTs=()

	for word in $IO
	do
		if  [[ ${word::1} == ">" ]] ; then
			OUTs+=("${word:1}")
		elif [[ ${word::1} == "<" ]] ; then
			INs+=("${word:1}")
		fi
	done
	$DEBUG && echo -e "INs:\t" ${INs[@]}
	$DEBUG && echo -e "OUTs:\t" ${OUTs[@]}


	for OUT in ${OUTs[@]} ; do
		# add .hdr to target all:
		sed -i "1 s/$/ ${OUT}.hdr/" $OUTF
		printf "${OUT}.cfl ${OUT}.hdr " >> $OUTF
	done
	printf "&:" >> $OUTF
	for IN in ${INs[@]} ; do
		printf " ${IN}.cfl ${IN}.hdr" >> $OUTF
	done
	printf "\n" >> $OUTF
	printf "\tbart %s\n\n" "${cmdl}" >> $OUTF

	$DEBUG && echo "---------------------------------"
done

echo "Done."

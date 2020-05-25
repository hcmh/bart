#!/bin/bash
set -euo pipefail
set -B
#set +e

INDIR="$1"

if [ $# -gt 2 ]; then
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

for hdr in ${INDIR}/*.hdr ; do

	CMDLs["$hdr"]=$(sed '4q;d' "$hdr")
	IOs["$hdr"]=$(sed '6q;d' "$hdr")
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
	$DEBUG && echo -eecho -e "OUTs:\t" ${OUTs[@]}


	if [[ ${#OUTs[@]} -gt 0 ]] ; then

		for OUT in ${OUTs[@]} ; do
			printf "${OUT}.cfl ${OUT}.hdr " >> $OUTF
		done
		printf "&:" >> $OUTF
		for IN in ${INs[@]} ; do
			printf " ${IN}.cfl ${IN}.hdr" >> $OUTF
		done
		printf "\n" >> $OUTF
		printf "\tbart %s\n" "${cmdl}" >> $OUTF
	else
		printf "${hdr} has no output. No rule created\n" >&2
		printf "FIXME: This should create a .PHONY target!\n" >&2
	fi

	$DEBUG && echo "---------------------------------"
done

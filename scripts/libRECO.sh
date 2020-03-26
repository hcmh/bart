#!/bin/bash
# Copyright 2020. Uecker Lab. University Medical Center Göttingen.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# 
# Authors:
# 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
# 
# a function library for reconstructions of multi-echo data
# 


# =====================================================
#    global variables
# =====================================================

export DEBUG_LEVEL=3

M_PI=$(echo "4*a(1)" | bc -l)

NSMP=1        #
FSMP=1        #
NSPK=1        #
NCOI=1        #
NECO=1        #
NMEA=1        #
NSLI=1        #
TURN=1        #
GRAD=0        #

TRAJ_NAME="mems_hy"  #

NMEA4RECO=1
NECO4RECO=1

GRIDSIZE=1

GDC=""
GDC_METHOD="ring"

RECO_TRAJ_STR=""
RECO_GDC_STR="-c"


MODEL=1


DEST="/tmp/"
TEMP=${DEST}_temp

# =====================================================
#    
# =====================================================
set_dest_dir()
{
	DEST="$1"
	TEMP=${DEST}_temp
}

# =====================================================
#    GDC method: ring or oppo
# =====================================================
set_gdc_method()
{
	GDC_METHOD="$1"

	echo "  > set GDC_METHOD: $GDC_METHOD"
}

# =====================================================
#    model
# =====================================================
set_model()
{
	MODEL="$1"

	case $MODEL in
		0)
			echo "  > model: WF"
			;;
		1)
			echo "  > model: WFR2S"
			;;
		2)
			echo "  > model: WF2R2S"
			;;
		3)
			echo "  > model: R2S"
			;;
		*)
			echo "  > model: unknown"
			;;
	esac
}

# =====================================================
#    
# =====================================================
set_meas_for_reco()
{
	NMEA4RECO="$1" # value to be set

	if [[ $NMEA4RECO -ge $NMEA || $NMEA4RECO -le 0 ]]; then
		let NMEA4RECO=NMEA
	fi
}

# =====================================================
#    
# =====================================================
set_echo_for_reco()
{
	NECO4RECO="$1" # value to be set

	if [[ $NECO4RECO -ge $NECO || $NECO4RECO -le 0 ]]; then
		let NECO4RECO=NECO
	fi
}


# =====================================================
#    
# =====================================================
OVERGRID=$(echo "scale=2; 3/2" | bc)

set_gridsize()
{
	OVERGRID=$1

	GRIDSIZE=$(echo $FSMP*$OVERGRID/1 | bc)
}


# =====================================================
#    convert golden angle in rad to golden index
# =====================================================

golden_rad2ind()
{
	GRAD="$1" # INPUT

	GIND="$2" # OUTPUT

	ACCU=$(echo "1/10000" | bc -l)
	GRATIO=$(echo "(sqrt(5)+1)/2" | bc -l)	

	CIND=1

	while [ $CIND -lt 20 ]; do
		CRAD=$(echo "$M_PI/($GRATIO+$CIND-1)" | bc -l)
		DIFF=$(echo "$GRAD-$CRAD" | bc -l)
		if (( $(echo "${DIFF#-} < $ACCU" | bc -l) )); then 
			let GIND=CIND
			echo "  > find golen_index (GIND) as $GIND given golden_angle (GRAD) as $GRAD"
			break
		fi
		let CIND=CIND+1
	done

	if [ $CIND -eq 20 ]; then
		GIND=0
		echo "  > can't find the correct golden angle. set golden_index (GIND) back to 0"
	fi
}


# =====================================================
#    read raw data
# =====================================================

datread()
{
	DATFILE="$1" # INPUT raw-data file
	DATMODE="$2" # INPUT data mode

	KDAT="$3" # OUTPUT k-space data


	echo "  > read raw k-space data"


	if [ "$DATMODE" = "MPI" ]; then

		# readin the "dimensions:" line
		DIMS=`tail -c+5 $DATFILE | head -n10 | grep "^dimensions: " | cut -d":" -f2`

		NSMP=`echo ${DIMS} | cut -d"x" -f2`
		NSPK=`echo ${DIMS} | cut -d"x" -f3`
		NCOI=`echo ${DIMS} | cut -d"x" -f5`
		# NECO=`echo ${DIMS} | cut -d"x" -f6` # FIXME
		NMEA=`echo ${DIMS} | cut -d"x" -f12`
		NSLI=`echo ${DIMS} | cut -d"x" -f13`

		BCMEAS=0

		# readin the "reco:" line
		PARS=`tail -c+5 $DATFILE | head -n10 | grep "^reco: " | cut -d":" -f2`
		IFS=' '
		read -ra ARRAY <<< "$PARS"
		for str in "${ARRAY[@]:1}"; do

			if [ "${str:0:2}" = "-i" ]; then 
				TURN=${str:2}
			fi

			if [ "${str:0:2}" = "-x" ]; then 
				NSMP=${str:2}
			fi

			if [ "${str:0:2}" = "-d" ]; then
				FSMP=${str:2}
			fi

			if [ "${str:0:2}" = "-e" ]; then
				NECO=${str:2}
			fi

			if [ "${str:0:2}" = "-a" ]; then
				GRAD=${str:2}
			fi

			if [ "${str:0:2}" = "-k" ]; then
				TRAJ_NAME=${str:2}
			fi

			if [ "${str:0:2}" = "-B" ]; then
				BCMEAS=${str:2}
			fi

		done

		bart twixread -x${NSMP} -y${NSPK} -c${NCOI} -n${NMEA} -b${BCMEAS} -s${NSLI} $DATFILE $KDAT

		echo "  > NSMP $NSMP; FSMP $FSMP; NSPK $NSPK; NCOI $NCOI; NMEA $NMEA; NSLI $NSLI; TRAJ_NAME: $TRAJ_NAME"
	fi
}


# =====================================================
#    coil compression
# =====================================================

coicomp()
{
	VC=$1      # INPUT number of virtual coils
	KDAT0="$2" # INPUT k-space data

	KDAT1="$3" # OUTPUT coil-compressed k-space data

	echo "  > coil compression"

	bart cc -A -p${VC} $KDAT0 $KDAT1
}


# =====================================================
#    separate spokes to spoke and echo dimensions
# =====================================================

dat_spk2eco()
{
	TRAJ_NAME="$1"   # INPUT trajectory name
	KDAT0="$2"  # INPUT k-space data

	KDAT1="$3"  # OUTPUT k-space data

	bart transpose 1 2 $KDAT0 ${TEMP}_10
	bart transpose 0 1 ${TEMP}_10 $KDAT1

	rm -rf ${TEMP}*

	if [ "${TRAJ_NAME:0:4}" = "mems" ]; then
		echo "  > split traversing spokes to echoes"
		let TURN=NMEA
		let NSPK=NSPK/NECO
		bart transpose 2 10 $KDAT1 ${TEMP}_spk2frm
		bart reshape $(bart bitmask 5 10) $NECO $NSPK ${TEMP}_spk2frm ${TEMP}_esep
		bart transpose 5 10 ${TEMP}_esep ${TEMP}_eord
		bart transpose 2  5 ${TEMP}_eord ${TEMP}_sord
		bart transpose 5 10 ${TEMP}_sord $KDAT1

		rm -rf ${TEMP}*

		# if [ $FlipEvenEco -eq 1 ]; then
		# 	echo "  > flip samples from even echoes"
		# 	local IECO=0
		# 	while [ $IECO -lt $NECO ]; do

		# 		bart slice 5 $IECO $kdat ${temp}_kdat_$IECO
		# 		if [ $(($IECO%2)) -eq 1 ]; then # even echoes
		# 			bart flip $(bart bitmask 1) ${temp}_kdat_$IECO ${temp}_kdat_flip_$IECO
		# 		else
		# 			bart scale 1 ${temp}_kdat_$IECO ${temp}_kdat_flip_$IECO
		# 		fi

		# 		let IECO=IECO+1
		# 	done
		# 	bart join 5 `seq -s" " -f "${temp}_kdat_flip_%g" 0 $(( $NECO-1 ))` $kdat

		# 	FLIP_STR="-F"

		# else
		# 	FLIP_STR=""

		# fi

		RECO_TRAJ_STR="-E -e$NECO"

	elif [ "${TRAJ_NAME:0:4}" = "turn" ]; then
		RECO_TRAJ_STR=""
	fi

}


# =====================================================
#    calculate trajectory
# =====================================================
traj()
{
	RECO_TRAJ_STR="$1" # INPUT trajectory string for traj
	RECO_GDC_STR="$2"  # INPUT gradient delay correction string

	TRAJ="$3"          # OUTPUT trajectory file

	echo "  > calculate trajectory with GDC: $RECO_GDC_STR"

	bart traj -x$NSMP -d$FSMP -y$NSPK -t$NMEA -r -s$GIND -D $RECO_TRAJ_STR $RECO_GDC_STR ${TEMP}_traj

	bart scale $OVERGRID ${TEMP}_traj $TRAJ

	rm -rf ${TEMP}*
}


# =====================================================
#    estimate gradient delay
# =====================================================

estdelay()
{
	ECOWISE="$1" # 

	KDAT="$2"    # INPUT kdat
	TRAJ="$3"    # INPUT traj

	GDC="$4"     # OUTPUT gradient delay correction file


	echo "  > estimate gradient delay coefficients"


	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $KDAT ${TEMP}_kdat_estdelay
	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $TRAJ ${TEMP}_traj_estdelay

	NSPK4GDC=400 # number of spokes for GDC

	bart resize -c 10 $NSPK4GDC ${TEMP}_kdat_estdelay ${TEMP}_kdat_estdelay_t
	bart resize -c 10 $NSPK4GDC ${TEMP}_traj_estdelay ${TEMP}_traj_estdelay_t

	bart transpose 2 10 ${TEMP}_kdat_estdelay_t ${TEMP}_kdat_estdelay_a
	bart transpose 2 10 ${TEMP}_traj_estdelay_t ${TEMP}_traj_estdelay_a



	bart zeros 16 3 1 1 1 1 $NECO 1 1 1 1 1 1 1 1 1 1 $GDC



	local TOTALECO=$(( ($ECOWISE==1) ? NECO : 1 ))

	local IECO=0
	while [ $IECO -lt $TOTALECO ]; do

		bart slice 5 $IECO ${TEMP}_kdat_estdelay_a ${TEMP}_kdat_estdelay_${IECO}
		bart slice 5 $IECO ${TEMP}_traj_estdelay_a ${TEMP}_traj_estdelay_${IECO}

		local LEN=$(( ($IECO%2 == 0) ? ($FSMP/2) : ($FSMP/2 + 2) ))

		if [ $(($IECO%2)) -eq 1 ]; then # even echoes
			bart flip $(bart bitmask 1) ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk
			bart flip $(bart bitmask 1) ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt
		else
			bart scale 1 ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk
			bart scale 1 ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt
		fi

		bart resize 1 $LEN ${TEMP}_kk ${TEMP}_kk_r
		bart resize 1 $LEN ${TEMP}_tt ${TEMP}_tt_r


		local GDC_OPT=$( [ "${GDC_METHOD:0:4}" == "ring" ] && echo "-R" || echo "" )


		bart estdelay $GDC_OPT ${TEMP}_tt_r ${TEMP}_kk_r ${TEMP}_GDC_$IECO

		let IECO=IECO+1
	done

	if [ $ECOWISE -eq 1 ]; then 
		bart join 5 `seq -s" " -f "${TEMP}_GDC_%g" 0 $(( $NECO-1 ))` $GDC
	elif [ $ECOWISE -eq 0 ]; then
		IECO=0
		while [ $IECO -lt $NECO ]; do
			bart copy 5 $IECO ${TEMP}_GDC_0 $GDC
			let IECO=IECO+1
		done
	fi


	RECO_GDC_STR=$( [ "${GDC_METHOD:0:4}" == "ring" ] && echo "-c " || echo "" )"-O -V $GDC"

	rm -rf ${TEMP}*
}


# =====================================================
#    self gating via ssafary
# =====================================================

ssafary()
{
	KDAT0="$1"    # INPUT kdat
	TRAJ0="$2"    # INPUT traj

	EOF="$3"      # OUTPUT EOF
	KDAT1="$4"    # OUTPUT sorted kdat
	TRAJ1="$5"    # OUTPUT sorted traj


	echo "  > self gating via ssafary"

	local IECO=0

	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $KDAT0 ${TEMP}_kk
	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $TRAJ0 ${TEMP}_tt

	bart slice 5 $IECO ${TEMP}_kk ${TEMP}_kk_e
	bart slice 5 $IECO ${TEMP}_tt ${TEMP}_tt_e

	local CTR=$(( $FSMP/2 - ($FSMP - $NSMP) ))

	# extract DC component
	bart extract 1 $CTR $(( $CTR + 1 )) ${TEMP}_kk_e ${TEMP}_kk_ec


	bart rmfreq ${TEMP}_tt_e ${TEMP}_kk_ec ${TEMP}_kk_ec_rmfreq


	bart transpose 2 10 ${TEMP}_kk_ec_rmfreq ${TEMP}_kc

	bart squeeze ${TEMP}_kc ${TEMP}_ac1
	bart scale -- -1i ${TEMP}_ac1 ${TEMP}_ac2

	bart creal ${TEMP}_ac1 ${TEMP}_ac_real
	bart creal ${TEMP}_ac2 ${TEMP}_ac_imag

	bart join 1 ${TEMP}_ac_real ${TEMP}_ac_imag ${TEMP}_ac



	bart ssa -w231 ${TEMP}_ac ${EOF}



	local RESPI0=0 # 2
	local RESPI1=1 # 3

	local CARDI0=2 # 0
	local CARDI1=3 # 0

	bart slice 1 $RESPI0 $EOF ${TEMP}_EOF_r0
	bart slice 1 $RESPI1 $EOF ${TEMP}_EOF_r1

	bart slice 1 $CARDI0 $EOF ${TEMP}_EOF_c0
	bart slice 1 $CARDI1 $EOF ${TEMP}_EOF_c1

	bart join 1 ${TEMP}_EOF_r{0,1} ${TEMP}_EOF_c{0,1} ${TEMP}_tmp0
	bart transpose 1 11 ${TEMP}_tmp0 ${TEMP}_tmp1
	bart transpose 0 10 ${TEMP}_tmp1 ${TEMP}_eof

	local RESPI=7
	local CARDI=1 # 20

	bart bin -r0:1 -R$RESPI -c2:3 -C$CARDI -a600 ${TEMP}_eof ${TEMP}_kk ${TEMP}_ksg
	bart bin -r0:1 -R$RESPI -c2:3 -C$CARDI -a600 ${TEMP}_eof ${TEMP}_tt ${TEMP}_tsg


	bart transpose 11 10 ${TEMP}_ksg $KDAT1
	bart transpose 11 10 ${TEMP}_tsg $TRAJ1


	rm -rf ${TEMP}*

}



# =====================================================
#    gridding
# =====================================================

# grid()
# {


# 	echo "  > grid P"
# 	bart ones 16 1 $NSMP $NSPK 1 1 $NECO4RECO 1 1 1 1 $TURN 1 1 1 1 1 ${TEMP}_pdat
# 	bart nufft -d $GRIDSIZE:$GRIDSIZE:1 -a $traj4reco ${TEMP}_pdat ${TEMP}_psf
# 	bart fft -u 3 ${TEMP}_psf $P

# 	echo "  > grid Y"
# 	local IMEA=0
# 	while [ $IMEA -lt $NMEA4RECO ]; do
# 		bart slice 10 $IMEA $kdat4reco ${temp}_Y
# 		bart slice 10 $(($IMEA%$TURN)) $traj4reco ${temp}_traj
# 		bart nufft -d $GRIDSIZE:$GRIDSIZE:1 -a ${temp}_traj ${temp}_Y ${temp}_img_$IMEA
# 		bart fft -u 3 ${temp}_img_$IMEA ${temp}_Y_$IMEA
# 		let IMEA=IMEA+1
# 	done
# 	bart join 10 `seq -s" " -f "${temp}_Y_%g" 0 $((TMEA-1))` $Y

# 	rm -rf ${temp}*
# }


# =====================================================
#    initialization
# =====================================================

init()
{
	MODEL="$1"  # INPUT model
	KDAT="$2"   # INPUT kdat
	TRAJ="$3"   # INPUT traj

	R_INIT="$4" # OUTPUT R_INIT

	echo "  > initialization"

	local IECO=3

	# local TMEAS=$(bart show -d 10 $KDAT)

	# NMEA4RECO=$(( ($TMEAS < $NMEA4RECO) ? $TMEAS : $NMEA4RECO ))

	local IMEA=10 # $(( ($TMEA <= 10) ? TMEA : 10 ))

	bart extract 5 0 $IECO 10 0 $IMEA $KDAT ${TEMP}_init_kdat
	bart extract 5 0 $IECO 10 0 $IMEA $TRAJ ${TEMP}_init_traj
	bart extract 5 0 $IECO $TE ${TEMP}_init_TE

	bart nufft -d $GRIDSIZE:$GRIDSIZE:1 -a ${TEMP}_init_traj ${TEMP}_init_kdat ${TEMP}_init_Yi

	bart slice 3 0 ${TEMP}_init_Yi ${TEMP}_dixon_I0
	bart avg $(bart bitmask 10) ${TEMP}_dixon_I0 ${TEMP}_dixon_I1
	bart dixon ${TEMP}_init_TE ${TEMP}_dixon_I1 ${TEMP}_dixon_WF

	# bart fft -u 3 ${TEMP}_init_Yi ${TEMP}_init_Y

	let NMAP=VC+3

	bart zeros 16 $GRIDSIZE $GRIDSIZE 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ${TEMP}_img0
	bart zeros 16 $GRIDSIZE $GRIDSIZE 1 1 1 1 $VC 1 1 1 1 1 1 1 1 1 ${TEMP}_coi0

	bart join 6 ${TEMP}_dixon_WF ${TEMP}_img0 ${TEMP}_coi0 ${TEMP}_dixon_R_INIT

	bart mobaT2star -O -M0 -R0 -i6 -r2 -a880 -d5 -g -F$FSMP -t ${TEMP}_init_traj ${TEMP}_init_kdat ${TEMP}_init_TE ${TEMP}_R_INIT ${TEMP}_C_INIT_$IMEA

	# bart mobaT2star -O -I ${TEMP}_dixon_R_INIT -M0 -R0 -i6 -r2 -a880 -d5 -g -t ${TEMP}_init_traj ${TEMP}_init_Y ${TEMP}_init_TE ${TEMP}_R_INIT # ${TEMP}_C_INIT_$IMEA

	bart extract 6 0 1 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_W_init
	bart extract 6 1 2 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_F_init
	bart extract 6 2 3 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_fB0_init

	# TODO: init coils as well
	# bart extract 10 $((IMEA-1)) $IMEA ${TEMP}_C_INIT_$IMEA ${TEMP}_C_INIT 

	case $MODEL in
		0)
			echo " __ init model 0: WF __"
			let NMAP=VC+3
			bart join 6 ${TEMP}_W_init ${TEMP}_F_init ${TEMP}_fB0_init ${TEMP}_coi0 $R_INIT
			;;

		1)
			echo " __ init model 1: WFR2S __"
			let NMAP=VC+4
			bart join 6 ${TEMP}_W_init ${TEMP}_F_init ${TEMP}_img0 ${TEMP}_fB0_init ${TEMP}_coi0 $R_INIT
			;;
		2)
			echo " __ init model 2: WF2R2S __"
			let NMAP=VC+5
			bart join 6 ${TEMP}_W_init ${TEMP}_img0 ${TEMP}_F_init ${TEMP}_img0 ${TEMP}_fB0_init ${TEMP}_coi0 $R_INIT
			;;
		3)
			echo " __ init model 3: R2S __"
			let NMAP=VC+3
			bart join 6 0 ${TEMP}_W_init ${TEMP}_img0 ${TEMP}_fB0_init ${TEMP}_coi0 $R_INIT
			;;
		*)
			echo " __ unknown model __"
			exit 1
			;;
	esac

	rm -rf ${TEMP}*

}



# =====================================================
#    reconstruction
# =====================================================

reco()
{
	KDAT="$1" # INPUT k-space data
	TRAJ="$2" # INPUT trajectory

	R="$3"    # OUTPUT reconstructed images



	local TMEAS=$(bart show -d 10 $KDAT)

	NMEA4RECO=$(( ($TMEAS < $NMEA4RECO) ? $TMEAS : $NMEA4RECO ))

	echo "  > iterative reconstructions on $NMEA4RECO frames"

	bart extract 10 0 $NMEA4RECO $KDAT ${TEMP}_KDAT4RECO
	bart extract 10 0 $NMEA4RECO $TRAJ ${TEMP}_TRAJ4RECO


	# NLINV

	local TECO=$(bart show -d 5 $KDAT)

	bart ones 16 1 1 1 1 1 $TECO 1 1 1 1 1 1 1 1 1 1 ${TEMP}_TE

	bart mobaT2star -M4 -R0 -i7 -r2 -T0.9 -a440 -d5 -g -o$OVERGRID -F$FSMP -t ${TEMP}_TRAJ4RECO ${TEMP}_KDAT4RECO ${TEMP}_TE $R

	rm -rf ${TEMP}*
}

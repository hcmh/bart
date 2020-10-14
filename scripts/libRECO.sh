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
FSMP=0        #
NSPK=1        #
NCOI=1        #
NECO=1        #
NMEA=1        #
NSLI=1        #
TURN=1        #
GRAD=0        #

TRAJ_NAME="mems_hy"  #

NMEA4RECO=0
NECO4RECO=1

GRIDSIZE=1

GDC=""
GDC_METHOD="ring"

RECO_TRAJ_STR=""
RECO_GDC_STR="-c"


MODEL=1

OS_SLI=0

DEST="/tmp/"
TEMP=${DEST}_temp

# =====================================================
#    
# =====================================================
set_dest_dir()
{
	DEST="$1"
	TEMP=${DEST}/_temp
}

# =====================================================
#    GDC method: ring or oppo
# =====================================================
set_gdc_method()
{
	GDC_METHOD="$1"

	echo "> set GDC_METHOD: $GDC_METHOD"
}

# =====================================================
#    model
# =====================================================
set_model()
{
	MODEL="$1"

	case $MODEL in
		0)
			echo "> model: WF"
			;;
		1)
			echo "> model: WFR2S"
			;;
		2)
			echo "> model: WF2R2S"
			;;
		3)
			echo "> model: R2S"
			;;
		*)
			echo "> model: unknown"
			;;
	esac
}

# =====================================================
#    
# =====================================================
set_full_sample()
{
	FSMP="$1"

	if [ $FSMP -lt $NSMP ] || [ $FSMP -le 0 ]; then
		let FSMP=NSMP
	fi
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
			echo "> find golen_index (GIND) as $GIND given golden_angle (GRAD) as $GRAD"
			break
		fi
		let CIND=CIND+1
	done

	if [ $CIND -eq 20 ]; then
		GIND=0
		echo "> can't find the correct golden angle. set golden_index (GIND) back to 0"
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


	echo "> read raw k-space data"


	if [ "$DATMODE" = "MPI" ]; then

		# readin the "dimensions:" line
		DIMS=`tail -c+5 $DATFILE | head -n10 | grep "^dimensions: " | cut -d":" -f2`

		NSMP=`echo ${DIMS} | cut -d"x" -f2`
		NSPK=`echo ${DIMS} | cut -d"x" -f3`
		NCOI=`echo ${DIMS} | cut -d"x" -f5`
		# NECO=`echo ${DIMS} | cut -d"x" -f6` # FIXME
		NMEA=`echo ${DIMS} | cut -d"x" -f12`
		NSLI=`echo ${DIMS} | cut -d"x" -f13` # FIXME

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

		golden_rad2ind $GRAD $GIND

	else

		bart twixread -A $DATFILE ${TEMP}_kdat0
		bart transpose 2 13 ${TEMP}_kdat0 $KDAT # move slices to the 13th dim

		NSMP=$(bart show -d  0 $KDAT)
		NSPK=$(bart show -d  1 $KDAT)
		NCOI=$(bart show -d  3 $KDAT)
		NMEA=$(bart show -d 10 $KDAT)
		NSLI=$(bart show -d 13 $KDAT)

	fi

	rm -rf ${TEMP}*
}


# =====================================================
#    coil compression
# =====================================================

coicomp()
{
	VC=$1      # INPUT number of virtual coils
	KDAT0="$2" # INPUT k-space data

	KDAT1="$3" # OUTPUT coil-compressed k-space data

	echo "> coil compression"

	bart cc -A -p${VC} $KDAT0 $KDAT1

	NCOI=$(bart show -d  3 $KDAT1)
}


# =====================================================
#    separate spokes to spoke and echo dimensions
# =====================================================

spk2eco()
{
	TRAJ_NAME="$1"   # INPUT trajectory name
	NECO="$2"
	OS_SLI="$3"      # INPUT oversampling in slice/partition direction
	KDAT0="$4"       # INPUT k-space data

	KDAT1="$5"       # OUTPUT k-space data

	bart transpose 1 2 $KDAT0 ${TEMP}_10
	bart transpose 0 1 ${TEMP}_10 $KDAT1

	rm -rf ${TEMP}*

	if [ "${TRAJ_NAME:0:4}" = "mems" ] && [ $NECO -gt 1 ]; then

		echo "> split traversing spokes to echoes"

		NMEA=$(bart show -d 10 $KDAT0)
		NTMP=$(bart show -d  1 $KDAT0)
		TURN=$(( NMEA ))
		NSPK=$(( NTMP / NECO ))

		bart reshape $(bart bitmask 2 5) $NECO $NSPK $KDAT1 ${TEMP}_kdat_es
		bart transpose 2 5 ${TEMP}_kdat_es $KDAT1

		rm -rf ${TEMP}*

		RECO_TRAJ_STR="-E -e$NECO"

	elif [ "${TRAJ_NAME:0:4}" = "turn" ]; then
		RECO_TRAJ_STR=""
	fi

	NSLI=$(bart show -d 13 $KDAT1)

	if [ $NSLI -gt 1 ]; then
		bart fft $(bart bitmask 13) $KDAT1 ${TEMP}_kdat1

		local CMP=`echo "$OS_SLI > 0" | bc`

		if [ $CMP -eq 1 ]; then
			NSLI=$(printf "%.0f" $(echo "scale=2; $NSLI/(1+$OS_SLI)" | bc))
		fi

		bart resize -c 13 $NSLI ${TEMP}_kdat1 $KDAT1
	fi

	NSLI=$(bart show -d 13 $KDAT1)

	rm -rf ${TEMP}*

}


# =====================================================
#    calculate trajectory
# =====================================================
traj()
{
	KDAT="$1"          # INPUT kdat
	FSMP="$2"
	GIND="$3"
	RECO_GDC_STR="$4"  # INPUT gradient delay correction string

	TRAJ="$5"          # OUTPUT trajectory file

	NSMP=$(bart show -d  1 $KDAT)
	NSPK=$(bart show -d  2 $KDAT)
	NMEA=$(bart show -d 10 $KDAT)
	NECO=$(bart show -d  5 $KDAT)
	NSLI=$(bart show -d 13 $KDAT)

	echo "> calculate trajectory with GDC: $RECO_GDC_STR"

	bart traj -x $NSMP -d $FSMP -y $NSPK -t $NMEA -m $NSLI -l -r -s $GIND -D -E -e $NECO $RECO_GDC_STR $TRAJ
}


# =====================================================
#    estimate gradient delay
# =====================================================

estdelay()
{
	ECOWISE="$1"      # INPUT echo wise correction (0 or 1)
	EVENECOSHIFT="$2" # INPUT even echo shift (2 - MPI;)
	KDAT="$3"         # INPUT kdat
	FSMP="$4"         # INPUT full sample
	TRAJ="$5"         # INPUT traj

	GDC="$6"          # OUTPUT gradient delay correction file


	NSMP=$(bart show -d  1 $KDAT)
	NSPK=$(bart show -d  2 $KDAT)
	NECO=$(bart show -d  5 $KDAT)
	NMEA=$(bart show -d 10 $KDAT)
	NSLI=$(bart show -d 13 $KDAT)

	local CTR_SLI_NR=$(( ($NSLI == 1) ? (0) : ( NSLI/2 ) ))

	echo "> estimate gradient delay coefficients using the central slice $CTR_SLI_NR"

	bart slice 13 $CTR_SLI_NR $KDAT ${TEMP}_kdat
	bart slice 13 $CTR_SLI_NR $TRAJ ${TEMP}_traj

	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) ${TEMP}_kdat ${TEMP}_kdat_estdelay
	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) ${TEMP}_traj ${TEMP}_traj_estdelay

	NSPK4GDC=80 # number of spokes for GDC

	bart resize -c 10 $NSPK4GDC ${TEMP}_kdat_estdelay ${TEMP}_kdat_estdelay_t
	bart resize -c 10 $NSPK4GDC ${TEMP}_traj_estdelay ${TEMP}_traj_estdelay_t

	bart transpose 2 10 ${TEMP}_kdat_estdelay_t ${TEMP}_kdat_estdelay_a
	bart transpose 2 10 ${TEMP}_traj_estdelay_t ${TEMP}_traj_estdelay_a



	bart zeros 16 3 1 1 1 1 $NECO 1 1 1 1 1 1 1 1 1 1 $GDC


	local FLIP_EVEN_ECO=1

	local TOTALECO=$(( ($ECOWISE==1) ? NECO : 1 ))

	local IECO=0
	while [ $IECO -lt $TOTALECO ]; do

		bart slice 5 $IECO ${TEMP}_kdat_estdelay_a ${TEMP}_kdat_estdelay_${IECO}
		bart slice 5 $IECO ${TEMP}_traj_estdelay_a ${TEMP}_traj_estdelay_${IECO}

		local CTR=$(( FSMP/2 ))
		local DIF=$(( FSMP - NSMP ))
		local RADIUS=$(( CTR - DIF ))


		# the echo position for even echoes are flipped
		local ECOPOS=$(( ($IECO%2 == 0) ? (RADIUS) : (RADIUS + EVENECOSHIFT - 1) ))
		local LEN=$(( ECOPOS * 2 ))

		if [ $FLIP_EVEN_ECO -eq 1 ]; then

			if [ $(($IECO%2)) -eq 1 ]; then # even echoes
				bart flip $(bart bitmask 1) ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk
				bart flip $(bart bitmask 1) ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt
			else
				bart scale 1 ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk
				bart scale 1 ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt
			fi

			bart resize 1 $LEN ${TEMP}_kk ${TEMP}_kk_r
			bart resize 1 $LEN ${TEMP}_tt ${TEMP}_tt_r

		else

			if [ $(($IECO%2)) -eq 1 ]; then # even echoes
				bart extract 1 $(( NSMP - (ECOPOS+1)*2 )) $NSMP ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk_r
				bart extract 1 $(( NSMP - (ECOPOS+1)*2 )) $NSMP ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt_r
			else
				bart resize 1 $LEN ${TEMP}_kdat_estdelay_${IECO} ${TEMP}_kk_r
				bart resize 1 $LEN ${TEMP}_traj_estdelay_${IECO} ${TEMP}_tt_r
			fi
		fi

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
	FSMP="$2"
	TRAJ0="$3"    # INPUT traj

	WINSIZE="$4"  # window size
	MOVAVG="$5"   # moving average

	EOF="$6"      # OUTPUT EOF
	SV="$7"       # OUTPUT SV
	KDAT1="$8"    # OUTPUT sorted kdat
	TRAJ1="$9"    # OUTPUT sorted traj

	NSMP=$(bart show -d  1 $KDAT0)
	NSPK=$(bart show -d  2 $KDAT0)
	NCOI=$(bart show -d  3 $KDAT0)
	NECO=$(bart show -d  5 $KDAT0)
	NMEA=$(bart show -d 10 $KDAT0)
	NSLI=$(bart show -d 13 $KDAT0)
	


	echo "> self gating via ssafary"

	local IECO=0

	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $KDAT0 ${TEMP}_kk
	bart reshape $(bart bitmask 2 10) 1 $(( NSPK * NMEA )) $TRAJ0 ${TEMP}_tt

	bart slice 5 $IECO ${TEMP}_kk ${TEMP}_kk_e
	bart slice 5 $IECO ${TEMP}_tt ${TEMP}_tt_e

	# bart slice 5 $IECO $GDC ${TEMP}_gd_e
	# bart cordelay -q ${TEMP}_gd_e ${TEMP}_tt_e ${TEMP}_kk_e ${TEMP}_kk_e_cor

	# extract DC component
	local CTR=$(( $FSMP/2 - ($FSMP - $NSMP) ))
	bart extract 1 $CTR $(( $CTR + 1 )) ${TEMP}_kk_e ${TEMP}_kk_ec


	bart rmfreq ${TEMP}_tt_e ${TEMP}_kk_ec ${TEMP}_kk_ec_rmfreq


	bart transpose 2 10 ${TEMP}_kk_ec_rmfreq ${TEMP}_kc_smp
	bart reshape $(bart bitmask 3 13) $(( NCOI*NSLI )) 1 ${TEMP}_kc_smp ${TEMP}_kc

	bart squeeze ${TEMP}_kc ${TEMP}_ac1
	bart scale -- -1i ${TEMP}_ac1 ${TEMP}_ac2

	bart creal ${TEMP}_ac1 ${TEMP}_ac_real
	bart creal ${TEMP}_ac2 ${TEMP}_ac_imag

	bart join 1 ${TEMP}_ac_real ${TEMP}_ac_imag ${TEMP}_ac



	bart ssa -w ${WINSIZE} ${TEMP}_ac ${EOF} ${SV}
	# ~/reco/bart_comp/bart ssa -w ${WINSIZE} ${TEMP}_ac ${EOF} ${SV}



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

	bart bin -r0:1 -R$RESPI -c2:3 -C$CARDI -a${MOVAVG} ${TEMP}_eof ${TEMP}_kk ${TEMP}_ksg
	bart bin -r0:1 -R$RESPI -c2:3 -C$CARDI -a${MOVAVG} ${TEMP}_eof ${TEMP}_tt ${TEMP}_tsg

	# no -a option when switching on -M
	# ~/reco/bart_comp/bart bin -M -r0:1 -R$RESPI -c2:3 -C$CARDI ${TEMP}_eof ${TEMP}_kk ${TEMP}_ksg
	# ~/reco/bart_comp/bart bin -M -r0:1 -R$RESPI -c2:3 -C$CARDI ${TEMP}_eof ${TEMP}_tt ${TEMP}_tsg

	bart transpose 11 10 ${TEMP}_ksg $KDAT1
	bart transpose 11 10 ${TEMP}_tsg $TRAJ1


	# rm -rf ${TEMP}*

}


# =====================================================
#    initialization
# =====================================================

init()
{
	MODEL="$1"  # INPUT model
	KDAT="$2"   # INPUT kdat
	TRAJ="$3"   # INPUT traj
	GRIDSIZE="$4"
	FSMP="$5"
	DIXON="$6"

	INITIMG="$7"

	R_INIT="$8" # OUTPUT R_INIT

	VC=$(bart show -d 3 $KDAT)

	echo "> initialization"

	NSLI=$(bart show -d 13 $KDAT)

	local IECO=3

	local TMEA=$(bart show -d 10 $KDAT)

	local IMEA=$(( ($TMEA <= 7) ? TMEA : 7 ))

	for (( S=0; S<${NSLI}; S++ )); do

		bart slice 13 $S $KDAT ${TEMP}_kdat0
		bart slice 13 $S $TRAJ ${TEMP}_traj0

		bart extract 5 0 $IECO 10 0 $IMEA ${TEMP}_kdat0 ${TEMP}_init_kdat
		bart extract 5 0 $IECO 10 0 $IMEA ${TEMP}_traj0 ${TEMP}_init_traj
		bart extract 5 0 $IECO $TE ${TEMP}_init_TE

		let NMAP=VC+3

		bart zeros 16 $GRIDSIZE $GRIDSIZE 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ${TEMP}_img0
		bart zeros 16 $GRIDSIZE $GRIDSIZE 1 1 1 1 $VC 1 1 1 1 1 1 1 1 1 ${TEMP}_coi0

		bart ones 16 $GRIDSIZE $GRIDSIZE 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ${TEMP}_img1
		bart scale 0.03 ${TEMP}_img1 ${TEMP}_img2

		bart scale 1 ${TEMP}_img0 ${TEMP}_R2S_init

		DIXON_INIT_STR=""
		if [ $DIXON -eq 1 ]; then

			bart nufft -d $GRIDSIZE:$GRIDSIZE:1 -a ${TEMP}_init_traj ${TEMP}_init_kdat ${TEMP}_init_Yi
			bart slice 3 0 ${TEMP}_init_Yi ${TEMP}_dixon_I0
			bart avg $(bart bitmask 10) ${TEMP}_dixon_I0 ${TEMP}_dixon_I1
			bart dixon ${TEMP}_init_TE ${TEMP}_dixon_I1 ${TEMP}_dixon_WF
			bart join 6 ${TEMP}_dixon_WF ${TEMP}_img0 ${TEMP}_coi0 ${TEMP}_dixon_R_INIT

			DIXON_INIT_STR="-I ${TEMP}_dixon_R_INIT"
		fi

		bart mobaT2star -O -M0 -i6 -R2 -F$FSMP ${DIXON_INIT_STR} -t ${TEMP}_init_traj ${TEMP}_init_kdat ${TEMP}_init_TE ${TEMP}_R_INIT

		bart extract 6 2 3 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_fB0_init

		if [ $INITIMG -eq 0 ]; then

			bart extract 6 0 1 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_W_init
			bart extract 6 1 2 10 $((IMEA-1)) $IMEA ${TEMP}_R_INIT ${TEMP}_F_init
		
		elif [ $INITIMG -eq 1 ]; then

			bart scale 0.1 ${TEMP}_img1 ${TEMP}_W_init
			bart scale 0.1 ${TEMP}_img1 ${TEMP}_F_init

		fi

		# TODO: init coils as well
		# bart extract 10 $((IMEA-1)) $IMEA ${TEMP}_C_INIT_$IMEA ${TEMP}_C_INIT 

		case $MODEL in
			0)
				echo " __ init model 0: WF __"
				let NMAP=VC+3
				bart join 6 ${TEMP}_W_init ${TEMP}_F_init ${TEMP}_fB0_init ${TEMP}_coi0 ${TEMP}_R_INIT_S${S}
				;;

			1)
				echo " __ init model 1: WFR2S __"
				let NMAP=VC+4
				bart join 6 ${TEMP}_W_init ${TEMP}_F_init ${TEMP}_R2S_init ${TEMP}_fB0_init ${TEMP}_coi0 ${TEMP}_R_INIT_S${S}
				;;
			2)
				echo " __ init model 2: WF2R2S __"
				let NMAP=VC+5
				bart join 6 ${TEMP}_W_init ${TEMP}_R2S_init ${TEMP}_F_init ${TEMP}_R2S_init ${TEMP}_fB0_init ${TEMP}_coi0 ${TEMP}_R_INIT_S${S}
				;;
			3)
				echo " __ init model 3: R2S __"
				let NMAP=VC+3
				bart join 6 ${TEMP}_W_init ${TEMP}_R2S_init ${TEMP}_fB0_init ${TEMP}_coi0 ${TEMP}_R_INIT_S${S}
				;;
			4)
				echo " __ init model 4: PHASEDIFF __"
				let NMAP=VC+2
				bart join 6 ${TEMP}_W_init ${TEMP}_fB0_init ${TEMP}_coi0 ${TEMP}_R_INIT_S${S}
				;;
			*)
				echo " __ unknown model __"
				exit 1
				;;
		esac

	done

	bart join 13 `seq -s" " -f "${TEMP}_R_INIT_S%g" 0 $(( $NSLI - 1))` $R_INIT

	rm -rf ${TEMP}*

}



# =====================================================
#    reconstruction
# =====================================================

reco()
{
	RECOTYPE="$1"   # NLINV, PICS, MOBA

	TRAJ="$2"       # INPUT trajectory
	KDAT="$3"       # INPUT k-space data
	TE="$4"         # INPUT echo-time file

	R="$5"          # OUTPUT reconstructed images

	# case "$RECOTYPE" in

	# 	"NLINV")

	# 		bart rtnlinv -r2 -d5 -g -t $TRAJ $KDAT $R
	# 		;

	# 	"PICS")

	# 		;
	# 	"MOBA")

	# 		bart mobaT2star -M2 -R0 -i7 -d5 -g -o$OVERGRID -F$FSMP -t ${TEMP}_TRAJ4RECO ${TEMP}_KDAT4RECO $TE $R
	# 		;

	# esac

	rm -rf ${TEMP}*
}

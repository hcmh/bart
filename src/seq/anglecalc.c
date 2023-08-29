#include "anglecalc.h"
#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef	M_GR
#define M_GR    (double) (sqrt(5.)+1)/2
#endif

//  ------------------------------------------------------------------
/// The following function dReturnRotAngle returns the incremental
/// Rotation angle phi that is needed for various radial sampling schemes
//  ------------------------------------------------------------------


bool bcalc_base_angles(double *angle_spoke, double* angle_frame, double* angle_slice,
			     enum ePEMode mode, long num_turns, int mb_factor, bool double_angle,
			     long lines_to_measure, long num_slices, long inv_repets)
{

	double GA = M_PI/(M_GR + num_turns - 1);
	double double_angle_factor = double_angle ? 2. : 1.;

	bool ret = true;
	switch (mode)
	{

		// Radial | Aligned frames | Aligned partitions
		case PEMODE_RAD_ALAL:

			*angle_spoke = double_angle_factor * M_PI / (double)lines_to_measure;
			*angle_frame = 0.;
			*angle_slice = 0.;

			break;

		// Radial | Turned frames | Aligned partitions
		case PEMODE_RAD_TUAL:

			*angle_spoke = double_angle_factor * M_PI / (double)lines_to_measure;
			*angle_frame = double_angle_factor * M_PI / (double)lines_to_measure / (double)num_turns;
			*angle_slice = 0.;

			break;

		// Radial | Turned frames | (Linear)-turned partitions
		case PEMODE_RAD_TUTU:

			*angle_spoke = double_angle_factor * M_PI / (double)lines_to_measure;
			*angle_frame = double_angle_factor * M_PI / (double)lines_to_measure / (double)num_turns / (double)mb_factor;
			*angle_slice = 1.        * M_PI / (double)lines_to_measure / (double)mb_factor;

			break;

		// Radial | Turned frames | Golden-angle partitions
		case PEMODE_RAD_TUGA:

			*angle_spoke = double_angle_factor * M_PI / (double)lines_to_measure;
			*angle_frame = double_angle_factor * M_PI / (double)lines_to_measure / (double)num_turns;
			*angle_slice = M_PI / (double)lines_to_measure / M_GR;
// 			*angle_slice = fmod( (double)lSlice * (M_PI / (double)lines_to_measure) / M_GR, M_PI / (double)lines_to_measure ); // FIXME

			break;

		// Radial | Golden-angle frames | Aligned partitions
		case PEMODE_RAD_GAAL:

			*angle_spoke = GA;
			*angle_frame = *angle_spoke * lines_to_measure;
			*angle_slice = 0.;

			break;

		// Radial | Consecutive spokes aquired in GA fashion
		case PEMODE_RAD_GA:

			*angle_slice = GA;
			*angle_spoke = *angle_slice * (double)( num_slices );
			*angle_frame = *angle_slice * (double)( num_slices * lines_to_measure );

			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_ALAL:

			*angle_spoke = double_angle_factor * M_PI / (double)inv_repets;
			*angle_frame = 0.;
			*angle_slice = 0.;

			break;

		// Radial | Multiple inversion recovery | Golden-angle partitions
		case PEMODE_RAD_MINV_GA:

			*angle_slice = GA;
			*angle_spoke = *angle_slice * (double)( num_slices );
			*angle_frame = *angle_slice * (double)( num_slices * lines_to_measure );

			break;

		// Radial | Multiple inversion recovery | Aligned partitions
		case PEMODE_RAD_MINV_GAAL:

			*angle_spoke = GA;
			*angle_frame = *angle_spoke * lines_to_measure;;
			*angle_slice = 0.;

			break;


		// Radial | Hybrid
		//   1) uniform spoke distribution within one frame
		//   2) golden angle increment between frames and partitions
		// Refer to:
		// * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
		// * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
		// * estimation using undersampled triple-echo multi-spoke radial FLASH.
		// * Magn Reson Med 82:1000-1011 (2019)
		//
		case PEMODE_RAD_MEMS_HYB:

			*angle_spoke = double_angle_factor * M_PI / (double)lines_to_measure;
			*angle_slice = 0.; // TODO: M_PI / ( M_GR + (double)total_turns - 1. );
			*angle_frame = GA; // *angle_slice * (double)( total_slices );

			break;


		case PEMODE_RAD_RAND:
		case PEMODE_RAD_RANDAL: // No long implemented! // Radial | Randomly aligned
		default:

			ret = false;
			break;
	}


	return ret;

}




double dgetRotAngle(
	long excitation,
	long echo,
	long repetition,
	long inversion_repetition,
	long slice,
	enum ePEMode mode,
	long num_slices,
	long num_turns,
	long num_echoes,
	long num_inv_repets,
	long mb_factor,
	long lines_to_measure,
	long repetitions_to_measure,
	long start_pos_GA,
	bool double_angle)
{

	double phi = 0.; // spoke angle to be returned [rad]

	double angle_spoke = 0.; // increment angle for spoke [rad]
	double angle_frame = 0.; // increment angle for frame [rad]
	double angle_slice = 0.; // increment angle for slice [rad]

	long line = get_spoke_index(mode, excitation, echo, num_echoes);


	bcalc_base_angles(&angle_spoke, &angle_frame, &angle_slice,
			mode, num_turns, mb_factor, double_angle,
		  lines_to_measure, num_slices, num_inv_repets);



	switch (mode)
	{

		// Radial | Aligned frames | Aligned partitions
		case PEMODE_RAD_ALAL:
		// Radial | Turned frames | Aligned partitions
		case PEMODE_RAD_TUAL:
		// Radial | Turned frames | (Linear)-turned partitions
		case PEMODE_RAD_TUTU:

			phi = angle_spoke * line
				+ angle_frame * (repetition % num_turns)
				+ angle_slice * slice;

			break;

		// Radial | Turned frames | Golden-angle partitions
		case PEMODE_RAD_TUGA:

			//Calc total rotation angle
			phi = angle_spoke * line
				+ angle_frame * (repetition % num_turns)
				+ fmod( slice * angle_slice, M_PI / lines_to_measure);

			break;

		// Radial | Golden-angle frames | Aligned partitions
		case PEMODE_RAD_GAAL:

			phi = angle_spoke * line
				+ angle_frame * (start_pos_GA + repetition);

			if (!double_angle)
				phi = fmod(phi, M_PI);

			break;

		// Radial | Consecutive spokes aquired in GA fashion
		case PEMODE_RAD_GA:

			phi = angle_slice * slice
				+ angle_spoke * line
				+ angle_frame * repetition;

			if (!double_angle)
				phi = fmod(phi, M_PI);

			break;

		// Radial | Multiple inversion recovery |
		case PEMODE_RAD_MINV_ALAL:

			phi = angle_spoke * inversion_repetition;

			break;

		// Radial | Multiple inversion recovery | Golden-angle partitions
		case PEMODE_RAD_MINV_GA:

			phi = angle_slice * slice
				+ angle_spoke * line
				+ angle_frame * ( repetition + inversion_repetition * (repetitions_to_measure+1) );

			if (!double_angle)
				phi = fmod(phi, M_PI);

			break;

		// Radial | Multiple inversion recovery | Aligned partitions
		 case PEMODE_RAD_MINV_GAAL:

			phi = angle_spoke * line
				+ angle_frame * ( start_pos_GA + repetition + inversion_repetition * (repetitions_to_measure+1) );

			if (!double_angle)
				phi = fmod(phi, M_PI);

			break;

		// Radial | Hybrid
		//   1) uniform spoke distribution within one frame
		//   2) golden angle increment between frames and partitions
		// Refer to:
		// * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
		// * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
		// * estimation using undersampled triple-echo multi-spoke radial FLASH.
		// * Magn Reson Med 82:1000-1011 (2019)
		//
		case PEMODE_RAD_MEMS_HYB:

			phi = angle_spoke * line
				+ angle_slice * slice
				+ angle_frame * repetition;

			break;


		case PEMODE_RAD_RAND:
		case PEMODE_RAD_RANDAL: // No long implemented! // Radial | Randomly aligned
		default:

			phi = 0.;
			break;
	}


	if ( PEMODE_RAD_MEMS_HYB != mode ) {

		if ( (echo % 2) == 1 )
			phi += M_PI;
	} else { // MEMS

		if ( ((echo % 2) == 1) && (angle_spoke < M_PI/2.) )
			phi += M_PI;
	}

	return phi;
}





long get_spoke_index( enum ePEMode mode, long lExcitation, long lEcho,long lEchoes)
{
	long lSpokeIndex = 0;

	if ( PEMODE_RAD_MEMS_HYB != mode )
		lSpokeIndex  = lExcitation;
	else
		lSpokeIndex  = lExcitation * lEchoes + lEcho;

	return lSpokeIndex;
}



double dgetRotAngle_ref(
	long lExcitation,
	long lEcho,
	long lRepetition,
	long lInvRepet,
	long lSlice,
	enum ePEMode mode,
	long lTotalSlices,
	long lTotalTurns,
	long lEchoes,
	long m_lInvRepets,
	long mbFactor,
	long m_lLinesToMeasure,
	long m_lRepetitionsToMeasure,
	long lStartPosGA,
	bool double_angle)
{

	double dPhi          = 0.; // spoke angle to be returned [rad]

	double dPhi_IncreSpk = 0.; // increment angle for spoke [rad]
	double dPhi_IncreFrm = 0.; // increment angle for frame [rad]
	double dPhi_IncreSlc = 0.; // increment angle for slice [rad]

	double dScalePhi     = double_angle ? 2. : 1.;

	long lLine           = get_spoke_index( mode, lExcitation, lEcho, lEchoes);





	switch (mode)
	{

		// Radial | Aligned frames | Aligned partitions
	case PEMODE_RAD_ALAL:

		dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lLinesToMeasure;
		dPhi_IncreFrm = 0.;
		dPhi_IncreSlc = 0.;

		dPhi = dPhi_IncreSpk * (double)lLine;

		break;

		 // Radial | Turned frames | Aligned partitions
	 case PEMODE_RAD_TUAL:

		 dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lLinesToMeasure;
		 dPhi_IncreFrm = dScalePhi * M_PI / (double)m_lLinesToMeasure / (double)lTotalTurns;
		 dPhi_IncreSlc = 0.;

		 dPhi = dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)(lRepetition % lTotalTurns);

		 break;

		 // Radial | Turned frames | (Linear)-turned partitions
	 case PEMODE_RAD_TUTU:

		 dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lLinesToMeasure;
		 dPhi_IncreFrm = dScalePhi * M_PI / (double)m_lLinesToMeasure / (double)lTotalTurns / (double)mbFactor;
		 dPhi_IncreSlc = 1.        * M_PI / (double)m_lLinesToMeasure / (double)mbFactor;

		 dPhi = dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)(lRepetition % lTotalTurns)
		 + dPhi_IncreSlc * (double)lSlice;

		 break;

		 // Radial | Turned frames | Golden-angle partitions
	 case PEMODE_RAD_TUGA:

		 dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lLinesToMeasure;
		 dPhi_IncreFrm = dScalePhi * M_PI / (double)m_lLinesToMeasure / (double)lTotalTurns;
		 dPhi_IncreSlc = fmod( (double)lSlice * (M_PI / (double)m_lLinesToMeasure) / M_GR, M_PI / (double)m_lLinesToMeasure ); // FIXME

		 //Calc total rotation angle
		 dPhi = dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)(lRepetition % lTotalTurns)
		 + dPhi_IncreSlc;

		 break;

	// Radial | Golden-angle frames | Aligned partitions
	case PEMODE_RAD_GAAL:

		dPhi_IncreSpk = M_PI / ( M_GR + (double)lTotalTurns - 1. );
		dPhi_IncreFrm = dPhi_IncreSpk * m_lLinesToMeasure;
		dPhi_IncreSlc = 0.;

		dPhi = dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)(lStartPosGA + lRepetition);

		if (!double_angle) {
			 dPhi = fmod(dPhi, M_PI);
		}

		break;

	// Radial | Consecutive spokes aquired in GA fashion
	case PEMODE_RAD_GA:

		dPhi_IncreSlc = M_PI / ( M_GR + (double)lTotalTurns - 1. );
		dPhi_IncreSpk = dPhi_IncreSlc * (double)( lTotalSlices );
		dPhi_IncreFrm = dPhi_IncreSlc * (double)( lTotalSlices * m_lLinesToMeasure );

		dPhi = dPhi_IncreSlc * (double)lSlice
		 + dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)lRepetition;

		if (!double_angle) {
			 dPhi = fmod(dPhi, M_PI);
		}

		break;

	// Radial | Multiple inversion recovery |
	case PEMODE_RAD_MINV_ALAL:

		 dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lInvRepets;
		 dPhi_IncreFrm = 0.;
		 dPhi_IncreSlc = 0.;

		 dPhi = dPhi_IncreSpk * (double)lInvRepet;

		 break;

		 // Radial | Multiple inversion recovery | Golden-angle partitions
	 case PEMODE_RAD_MINV_GA:

		 dPhi_IncreSlc = M_PI / ( M_GR + (double)lTotalTurns - 1. );
		 dPhi_IncreSpk = dPhi_IncreSlc * (double)( lTotalSlices );
		 dPhi_IncreFrm = dPhi_IncreSlc * (double)( lTotalSlices * m_lLinesToMeasure );

		 dPhi = dPhi_IncreSlc * (double)lSlice
		 + dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)( lRepetition + lInvRepet * (m_lRepetitionsToMeasure+1) );

		 if (!double_angle){
			 dPhi = fmod(dPhi, M_PI);
			 }

		break;

			 // Radial | Multiple inversion recovery | Aligned partitions
	 case PEMODE_RAD_MINV_GAAL:

		 dPhi_IncreSpk = M_PI / ( M_GR + (double)lTotalTurns - 1. );
		 dPhi_IncreFrm = dPhi_IncreSpk * m_lLinesToMeasure;
		 dPhi_IncreSlc = 0.;

		 dPhi = dPhi_IncreSpk * (double)lLine
		 + dPhi_IncreFrm * (double)( lStartPosGA + lRepetition + lInvRepet * (m_lRepetitionsToMeasure+1) );

		 if (!double_angle){
			 dPhi = fmod(dPhi, M_PI);
			 }

		break;


	// Radial | Hybrid
	//   1) uniform spoke distribution within one frame
	//   2) golden angle increment between frames and partitions
	// Refer to:
	// * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
	// * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
	// * estimation using undersampled triple-echo multi-spoke radial FLASH.
	// * Magn Reson Med 82:1000-1011 (2019)
	//
	case PEMODE_RAD_MEMS_HYB:

		dPhi_IncreSpk = dScalePhi * M_PI / (double)m_lLinesToMeasure;
		dPhi_IncreSlc = 0.; // TODO: M_PI / ( M_GR + (double)lTotalTurns - 1. );
		dPhi_IncreFrm = M_PI / ( M_GR + (double)lTotalTurns - 1. ); // dPhi_IncreSlc * (double)( lTotalSlices );

		dPhi = dPhi_IncreSpk * (double)lLine
		+ dPhi_IncreSlc * (double)lSlice
		+ dPhi_IncreFrm * (double)lRepetition;

		break;


	case PEMODE_RAD_RAND:
	case PEMODE_RAD_RANDAL: // No long implemented! // Radial | Randomly aligned
	default:

		dPhi = 0.;
		break;
	}


	if ( PEMODE_RAD_MEMS_HYB != mode )
	{
		if ( (lEcho%2) == 1 )
			dPhi += M_PI;
	}
	else // MEMS
	{
		if ( ((lEcho%2) == 1) && (dPhi_IncreSpk < M_PI/2.) )
			dPhi += M_PI;
	}

	return dPhi;

}

#ifndef __ANGLE_CALC_H
#define __ANGLE_CALC_H


enum ePEMode
{
	// PEMODE_CARTESIAN = 1,
	PEMODE_RAD_ALAL = 1,
	PEMODE_RAD_TUAL,
	PEMODE_RAD_GAAL,
	PEMODE_RAD_GA,
	PEMODE_RAD_TUGA,
	PEMODE_RAD_TUTU,
	PEMODE_RAD_RANDAL,
	PEMODE_RAD_RAND,
	PEMODE_RAD_MINV_ALAL,
	PEMODE_RAD_MINV_GA,
	PEMODE_RAD_MINV_GAAL,
	PEMODE_RAD_MEMS_HYB
};

#ifndef __cplusplus
#ifndef bool
typedef int bool;
enum { false, true };
#endif
#endif

long get_spoke_index( enum ePEMode mode, long lExcitation, long lEcho,long lEchoes);


bool bcalc_base_angles(double *angle_spoke, double* angle_frame, double* angle_slice,
		      enum ePEMode mode, long num_turns, int mb_factor, bool double_angle,
		      long lines_to_measure, long num_slices, long inv_repets);

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
	bool double_angle);

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
	bool double_angle);



#endif // __ANGLE_CALC_H

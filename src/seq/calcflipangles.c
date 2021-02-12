#include "calcflipangles.h"
#include "fascalings1.h"


double get_flipanglescaling( enum eContrast contrast, long index, bool preppulse)
{
	double beta = 0.5;
    switch (contrast) {
			
	case CONTRAST_RF_RANDOM:
	case CONTRAST_RF_SPOILED:
		return 1;
		break;

	case CONTRAST_BALANCED:
		if (preppulse)
			return 0.5;
		else
			return 1.0;
		break;

	case CONTRAST_LOVABLE:
		assert(index < N_ANGLES);
		/* return fascalings1[index]; */
		return 0.0;
		break;

	case CONTRAST_DABALANCED:
		// reference: Absil et al. MRM 2006
		// alpha -> alpha + a -> alpha -> alpha - a
		// so scalings: 1 -> 1 + beta -> 1 -> 1 - beta
		if (preppulse)
			return 0.5 - beta / 2.0;
		else
			switch (index%4) {

			case 1: return 1.0 + beta; break;
			case 3: return 1.0 - beta; break;
			default: return 1.0; break;
			}
		break;

	default: return 1.0;
	}			
}

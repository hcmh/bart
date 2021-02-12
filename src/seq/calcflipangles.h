#ifndef __CALCFLIPANGLES_H
#define __CALCFLIPANGLES_H

enum eContrast
{
	CONTRAST_RF_RANDOM = 1,
	CONTRAST_RF_SPOILED,
	CONTRAST_BALANCED,
	CONTRAST_LOVABLE,
	CONTRAST_DABALANCED,
};

#ifndef __cplusplus
#ifndef bool
typedef int bool;
enum { false, true };
#endif
#endif

double get_flipanglescaling( enum eContrast contrast, long index, bool preppulse);

#endif // __CALCFLIPANGLES_H

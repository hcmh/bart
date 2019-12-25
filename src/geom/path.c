
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <ctype.h>

#include "geom/draw.h"
#include "geom/logo.h"

#include "misc/debug.h"
#include "misc/misc.h"



const char* p1[] = {	// paths from inkscape, stroke-to-path
	"M 10.03632,-34.104001 H 1.9488 V 0 h 8.42856 c 5.3592,0 8.13624,-2.8257601 8.13624,-8.0875202 V -10.86456 c 0,-3.60528 -1.12056,-6.13872 -3.99504,-7.25928 v -0.09744 c 2.38728,-1.12056 3.45912,-3.312961 3.45912,-6.820801 v -1.218 c 0,-5.26176 -2.38728,-7.84392 -7.94136,-7.84392 z M 9.6952802,-15.3468 c 2.4359998,0 3.4591198,0.9744 3.4591198,4.1412 v 2.9719198 c 0,2.5334401 -0.9744,3.3616801 -2.77704,3.3616801 H 7.3080002 V -15.3468 Z m 0.19488,-13.885201 c 1.9000798,0 2.7283198,1.07184 2.7283198,3.50784 v 1.90008 c 0,2.72832 -1.218,3.60528 -3.2155198,3.60528 h -2.09496 v -9.0132 z",
	"M 33.62898,-34.104001 H 25.78506 L 20.32842,0 h 4.92072 l 0.92568,-6.1874402 h 6.5772 L 33.6777,0 h 5.40792 z m -4.23864,6.04128 h 0.09744 l 2.53344,17.246881 H 26.8569 Z",
	"M 58.19604,0 C 57.66012,-1.26672 57.6114,-2.4847201 57.6114,-4.1412001 v -5.2617601 c 0,-3.5565598 -0.87696,-6.0899998 -3.556561,-7.2105598 v -0.09744 c 2.387281,-1.12056 3.507841,-3.361681 3.507841,-6.869521 v -2.6796 c 0,-5.26176 -2.387281,-7.84392 -7.941361,-7.84392 h -8.08752 V 0 h 5.3592 v -13.8852 h 1.85136 c 2.436,0 3.50784,1.16928 3.50784,4.3360798 v 5.3592001 c 0,2.7770401 0.19488,3.31296008 0.4872,4.1899201 z m -8.720881,-29.232001 c 1.90008,0 2.72832,1.07184 2.72832,3.50784 v 3.36168 c 0,2.728321 -1.218,3.605281 -3.21552,3.605281 h -2.09496 v -10.474801 z",
	"m 59.576947,-29.232001 h 5.6028 V 0 h 5.3592 v -29.232001 h 5.602801 v -4.872 H 59.576947 Z",
	"m 2.0605469,1.5 v 21.220703 h 5 V 6.5 H 20.699219 v -5 z",
	"M 88.720703,33.839844 V 50.050781 H 75.080078 v 5 H 93.720703 V 33.839844 Z",
};
	
/* M X Y  move to X Y (m is relative)
 * L X Y line to X Y
 * H x	horizental
 * V y  vertical
 * Z close path
 * C X1 Y1, X2 Y2, X Y (c1 c1 end
 * S X2 Y2, X Y (short cut version where X1 Y1 is reflection)
 * Q X1 X2, X Y (quadratic one control point to both v)
 * T X Y short cut
 * A ...
 */

static int path2splines(int N, double p[N][2][4], const char* str)
{
	char c;
	int pos = 0;
	int off = 0;
	int l = 0;

	void xspline(float a[2], float c0[2], float c1[2], float b[2])
	{
		p[l][0][0] = a[0];
		p[l][1][0] = a[1];
		p[l][0][1] = c0[0];
		p[l][1][1] = c0[1];
		p[l][0][2] = c1[0];
		p[l][1][2] = c1[1];
		p[l][0][3] = b[0];
		p[l][1][3] = b[1];
		l++;
	}

	void xline(float a[2], float b[2])
	{
		xspline(a, a, b, b);
	}

	float old[2] = { 0., 0. };
	float cur[2] = { 0., 0. };
	float ini[2] = { 0., 0. };
	
	do {
		if (1 != sscanf(str + pos, " %c%n", &c, &off))
			break;

		pos += off;

		float x, y, x1, y1, x2, y2;

		x = cur[0];
		y = cur[1];

		if (isupper(c)) {

			cur[0] = 0.;
			cur[1] = 0.;
		}

		switch (c) {

		case 'M':
		case 'm':
			if (2 != sscanf(str + pos, "%f,%f%n", &x, &y, &off))
				goto err;

			pos += off;
			cur[0] += x;
			cur[1] += y;

			old[0] = cur[0]; old[1] = cur[1];
			ini[0] = cur[0]; ini[1] = cur[1];
			break;

		case 'v':
			x = 0.;
		case 'V':

			if (1 != sscanf(str + pos, "%f%n", &y, &off))
				goto err;

			goto line;
	
		case 'h':
			y = 0.;
		case 'H':

			if (1 != sscanf(str + pos, "%f%n", &x, &off))
				goto err;

			goto line;

		case 'L':
		case 'l':
			if (2 != sscanf(str + pos, "%f,%f%n", &x, &y, &off))
				goto err;
		line:
			pos += off;
			cur[0] += x;
			cur[1] += y;

			xline(old, cur);

			old[0] = cur[0]; old[1] = cur[1];
			break;

		case 'T':
		case 't':
			if (2 != sscanf(str + pos, "%f,%f %n", &x, &y, &off))
				goto err;

			// x1, y1, x2, y2 can be computed

			goto spline;
	
		case 'Q':
		case 'q':
			if (6 != sscanf(str + pos, "%f,%f %f,%f%n", &x2, &y2, &x, &y, &off))
				goto err;

			goto spline;

		case 'S':
		case 's':
			if (4 != sscanf(str + pos, "%f,%f %f,%f%n", &x2, &y2, &x, &y, &off))
				goto err;

			// x1, y1 from reflection of old

			goto spline;

		case 'C':
		case 'c':
			if (6 != sscanf(str + pos, "%f,%f %f,%f %f,%f%n", &x1, &y1, &x2, &y2, &x, &y, &off))
				goto err;

		spline:
			pos += off;

			float c0[2] = { x1 + cur[0], y1 + cur[1] };
			float c1[2] = { x2 + cur[0], y2 + cur[1] };

			cur[0] += x;
			cur[1] += y;

			xspline(old, c0, c1, cur);

			old[0] = cur[0]; old[1] = cur[1];
			break;
	
		case 'z':
		case 'Z':

			cur[0] = ini[0];
			cur[1] = ini[1];

			xline(old, cur);

			break;

		default:
			goto err;
		};

	} while(true);

	return l;
err:	
	return 0;
}


void *x = path2splines;
#if 0
#if 0
int main()
{
	double p[50][2][4];

	for (int j = 0; j < 6; j++) {

		int n = path2splines(50, p, p1[j]);

		int xx = 0;
		int yy = 0;

		if (j < 4) {
			xx += 9.98;
	        	yy += 45.87;
		}


		for (int i = 0; i < n; i++) {

			double coeff[2][4];

			coeff[0][0] = 4. * (p[i][1][0] + yy);
			coeff[1][0] = 4. * (p[i][0][0] + xx);
			coeff[0][2] = 4. * (p[i][1][3] + yy);
			coeff[1][2] = 4. * (p[i][0][3] + xx);
			coeff[0][1] = 4. * (p[i][1][1] - p[i][1][0]) * 3.;
			coeff[1][1] = 4. * (p[i][0][1] - p[i][0][0]) * 3.;
			coeff[0][3] = 4. * -(p[i][1][2] - p[i][1][3]) * 3.;
			coeff[1][3] = 4. * -(p[i][0][2] - p[i][0][3]) * 3.;

			printf("{ { %f, %f, %f, %f }, { %f, %f, %f, %f } },\n", 
					coeff[0][0], coeff[0][1], coeff[0][2], coeff[0][3],
					coeff[1][0], coeff[1][1], coeff[1][2], coeff[1][3]);
		}

		printf("----------\n");
	}
}
#else
int main()
{
	int X = 230;
	int Y = 385;
	complex float out[X][Y];

	for (int x = 0; x < X; x++)
		for (int y = 0; y < Y; y++)
			out[x][y] = 0.;

	
	for (int i = 0; i < ARRAY_SIZE(bart_logo); i++)
		cspline_cmplx(X, Y, &out, 1., bart_logo[i]);

	dump_cfl("out", 2, (long[]){ Y, X }, &out[0][0]);
}
#endif
#endif

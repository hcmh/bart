#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "sim_para.h"

const struct tissue_para tissue_ref_para[MAX_REF_VALUES] = {
    {      3.,     1.,  1.},
    {   0.877,  0.048,  1.},
    {   1.140,  0.06,   1.},
    {   1.404,  0.06,   1.},
    {   0.866,  0.095,  1.},
    {   1.159,  0.108,  1.},
    {   1.456,  0.122,  1.},
    {   0.883,  0.129,  1.},
    {   1.166,  0.150,  1.},
    {   1.442,  0.163,  1.}
};

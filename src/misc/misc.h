/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * Copyright 2017-2018. Damien Nguyen
 * Copyright 2019. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _MISC_H
#define _MISC_H

#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdnoreturn.h>

#include "misc/nested.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define MIN(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x < __y) ? __x : __y; })
#define MAX(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x > __y) ? __x : __y; })
#define CLAMP(x, min, max) MAX(min, MIN(x, max))

#define MAKE_ARRAY(x, ...) ((__typeof__(x)[]){ x, __VA_ARGS__ })
#define ARRAY_SIZE(x)	(sizeof(x) / sizeof(x[0]))

#define unreachable() __builtin_trap()

#define SWAP(x, y) do { __auto_type temp = x; x = y; y = temp; } while (0)

#include "misc/cppwrap.h"

#ifndef alloc_size
#ifndef __clang__
#define alloc_size(x) [[gnu::alloc_size(x)]]
#else
#define alloc_size(x)
#endif
#endif

extern void* xmalloc(size_t s) alloc_size(1);
extern void xfree(const void*);
extern void warn_nonnull_ptr(void*);

#define XMALLOC(x)	(x = xmalloc(sizeof(*x)))
#define XFREE(x)	(xfree(x), x = NULL)

#define _TYPE_ALLOC(T)		((T*)xmalloc(sizeof(T)))
#define TYPE_ALLOC(T)		_TYPE_ALLOC(__typeof__(T))
// #define TYPE_CHECK(T, x)	({ T* _ptr1 = 0; __typeof(x)* _ptr2 = _ptr1; (void)_ptr2; (x);  })

#ifndef __clang__
#define _PTR_ALLOC(T, x)										\
	T* x __attribute__((cleanup(warn_nonnull_ptr))) = xmalloc(sizeof(T))
#else
#define _PTR_ALLOC(T, x)										\
	T* x = xmalloc(sizeof(T))
#endif

#define _CONCAT(A, B) A ## B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(x) # x
#define STRINGIFY(x) _STRINGIFY(x)


#define PTR_ALLOC(T, x)		_PTR_ALLOC(__typeof__(T), x)
#define PTR_FREE(x)		XFREE(x)
#define PTR_PASS(x)		({ __typeof__(x) __tmp = (x); (x) = NULL; __tmp; })

#define ARR_CLONE(T, x)		({ PTR_ALLOC(T, __tmp2); memcpy(*__tmp2, x, sizeof(T)); *PTR_PASS(__tmp2); })

extern int parse_cfl(_Complex float res[1], const char* str);
extern int parse_double(double res[1], const char* str);
extern int parse_long(long res[1], const char* str);
extern int parse_longlong(long long res[1], const char* str);
extern int parse_ulonglong(unsigned long long res[1], const char* str);
extern int parse_int(int res[1], const char* str);
#ifndef __cplusplus
extern noreturn void error(const char* str, ...);
#else
extern __attribute__((noreturn)) void error(const char* str, ...);
#endif


#ifdef USE_DWARF
#undef assert
#define assert(expr) do { if (!(expr)) error("Assertion '" #expr "' failed in %s:%d\n",  __FILE__, __LINE__); } while (0)
#endif

extern int error_catcher(int fun(int argc, char* argv[__VLA(argc)]), int argc, char* argv[__VLA(argc)]);

extern int bart_printf(const char* fmt, ...) __attribute__((format(printf,1,2)));

extern void debug_print_bits(int dblevel, int D, unsigned long bitmask);

extern void print_dims(int D, const long dims[__VLA(D)]);
extern void debug_print_dims(int dblevel, int D, const long dims[__VLA(D)]);

#ifdef REDEFINE_PRINTF_FOR_TRACE
#define debug_print_dims(...) \
	debug_print_dims_trace(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)
#endif

extern void debug_print_dims_trace(const char* func_name,
				   const char* file,
				   int line,
				   int dblevel,
				   int D,
				   const long dims[__VLA(D)]);

typedef int CLOSURE_TYPE(quicksort_cmp_t)(int a, int b);

extern void quicksort(int N, int ord[__VLA(N)], quicksort_cmp_t cmp);

extern float quickselect(float *arr, int n, int k);
extern float quickselect_complex(_Complex float *arr, int n, int k);

extern void print_long(int D, const long arr[__VLA(D)]);
extern void print_float(int D, const float arr[__VLA(D)]);
extern void print_int(int D, const int arr[__VLA(D)]);
extern void print_complex(int D, const _Complex float arr[__VLA(D)]);

extern int bitcount(unsigned long flags);

extern const char* command_line;
extern char* stdin_command_line;
extern void* save_command_line(int argc, char* argv[__VLA(argc)]);

extern _Bool safe_isnanf(float x);
extern _Bool safe_isfinite(float x);

extern long io_calc_size(int D, const long dims[__VLA(D?:1)], size_t size);

extern char* ptr_printf(const char* fmt, ...) __attribute__((format(printf,1,2)));
extern char* ptr_vprintf(const char* fmt, va_list ap);
extern char* ptr_print_dims(int D, const long dims[__VLA(D)]);

extern char* construct_filename(int D, const long loopdims[__VLA(D)], const long pos[__VLA(D)], const char* prefix, const char* ext);


#define DEG2RAD(d) ((d) * M_PI / 180.)
#define RAD2DEG(r) ((r) / M_PI * 180.)


#include "misc/cppwrap.h"

#endif // _MISC_H


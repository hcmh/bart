
#include "misc/cppwrap.h"

extern int write_ra(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, unsigned int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, unsigned int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, unsigned int D, long dimensions[__VLA(D)]);

extern int write_multi_cfl_header(int fd, long num_ele, unsigned int D, unsigned int n[D], const long* dimensions[D]);
extern int read_multi_cfl_header(int fd, unsigned int D_max, unsigned int n_max, unsigned int n[D_max], long dimensions[D_max][n_max]);

extern void io_register_input(const char* name);
extern void io_register_output(const char* name);
extern void io_unregister(const char* name);

extern void io_memory_cleanup(void);

#include "misc/cppwrap.h"

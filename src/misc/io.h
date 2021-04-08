
#include "misc/cppwrap.h"


enum file_types_e {
	FILE_TYPE_CFL, FILE_TYPE_RA, FILE_TYPE_COO, FILE_TYPE_SHM,
#ifdef USE_MEM_CFL
	FILE_TYPE_MEM,
#endif
};

extern enum file_types_e file_type(const char* name);

extern int write_ra(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_ra(int fd, int n, long dimensions[__VLA(n)]);

extern int write_coo(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(int fd, int D, long dimensions[__VLA(D)]);

extern int write_multi_cfl_header(int fd, long num_ele, unsigned int D, unsigned int n[D], const long* dimensions[D]);
extern int read_multi_cfl_header(int fd, unsigned int D_max, unsigned int n_max, unsigned int n[D_max], long dimensions[D_max][n_max]);

extern void io_register_input(const char* name);
extern void io_register_output(const char* name);
extern void io_unregister(const char* name);

extern void io_unlink_if_opened(const char* name);

extern void io_memory_cleanup(void);

#include "misc/cppwrap.h"

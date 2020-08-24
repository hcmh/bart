


extern void memcache_off(void);
extern void memcache_clear(int device, void (*device_free)(const void* x));
extern _Bool mem_ondevice(const void* ptr);
extern _Bool mem_ondevice_num(const void* ptr, const int device);
extern _Bool mem_device_accessible(const void* ptr);
extern void mem_device_free(void* ptr, void (*device_free)(const void* x));
extern void* mem_device_malloc(int device, long size, void* (*device_alloc)(size_t));

extern void print_mem(void);
extern void print_memcache(void);




extern int png_write_rgb24(const char* name, int w, int h, long inum, const unsigned char* buf);
extern int png_write_rgb32(const char* name, int w, int h, long inum, const unsigned char* buf);
extern int png_write_bgr24(const char* name, int w, int h, long inum, const unsigned char* buf);
extern int png_write_bgr32(const char* name, int w, int h, long inum, const unsigned char* buf);

extern void construct_filename(unsigned int bufsize, char* name, unsigned int D, const long loopdims[D], const long pos[D], const char* prefix, const char* ext);

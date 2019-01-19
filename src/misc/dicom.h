
extern int dicom_write(const char* name, const char study_uid[64], const char series_uid[64], unsigned int cols, unsigned int rows, long inum, const unsigned char* img);
extern unsigned char* dicom_read(const char* name, int dims[2]);
extern void dicom_generate_uid(char buf[64]);



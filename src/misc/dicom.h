
#include <stdint.h>

extern int dicom_write(const char* name, const char study_uid[64], const char series_uid[64], unsigned int cols, unsigned int rows, long inum, const unsigned char* img);
extern unsigned char* dicom_read(const char* name, int dims[2]);
extern void dicom_generate_uid(char buf[64]);

struct dicom_obj_s;
extern struct dicom_obj_s* dicom_open(const char* name);
extern void dicom_close(const struct dicom_obj_s* dobj);
extern unsigned char* dicom_read_image(const struct dicom_obj_s* dobj, int dims[2]);
extern int dicom_instance_num(const struct dicom_obj_s* dobj);
extern void dicom_geometry(const struct dicom_obj_s* dobj, float pos[3][3]);

struct tag {

	uint16_t group;
	uint16_t element;
};

struct element {

	struct tag tag;
	char vr[2];

	unsigned int len;
	const void* data;
};

extern int dicom_query_tags(const struct dicom_obj_s* dobj, int N, struct element ellist[N]);



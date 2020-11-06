
typedef struct cJSON cJSON;
#include "simu/shepplogan.h"

extern char* readfile(const char *filename);

extern void read_json_to_struct(int N, struct ellipsis_s phantom[N], const cJSON* json_object);

extern void check_json_file(cJSON *json_file);

extern _Bool cJSON_not_NULL(const cJSON* object, const char* key);
extern int cJSON_get_int(const cJSON* object, const char* key);
extern float cJSON_get_float(const cJSON* object, const char* key);
extern _Bool cJSON_get_bool(const cJSON* object, const char* key);

extern void cJSON_set_long(long* dst, const cJSON* object, const char* key);
extern void cJSON_set_uint(unsigned int* dst, const cJSON* object, const char* key);
extern void cJSON_set_ulong(unsigned long* dst, const cJSON* object, const char* key);
extern void cJSON_set_int(int* dst, const cJSON* object, const char* key);
extern void cJSON_set_bool(_Bool* dst, const cJSON* object, const char* key);
extern void cJSON_set_float(float* dst, const cJSON* object, const char* key);

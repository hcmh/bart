
#include <stdbool.h>


#include "misc/cppwrap.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/cJSON.h"

typedef _Bool opt_json_f(void* ptr, const char* label, const cJSON* json_data);

struct opt_json_s {

	int n_labels;
	const char** label;
	opt_json_f* fun;
	void* ptr;
	const char* descr;
	_Bool needed;
};

extern opt_json_f opt_json_bool;
extern opt_json_f opt_json_int;
extern opt_json_f opt_json_uint;
extern opt_json_f opt_json_long;
extern opt_json_f opt_json_float;
extern opt_json_f opt_json_string;

#define JSON_LABEL(...) __VA_ARGS__

#define JSON_BOOL(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_bool, TYPE_CHECK(bool*, (ptr)), descr, needed}
#define JSON_UINT(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_uint, TYPE_CHECK(uint*, (ptr)), descr, needed}
#define JSON_INT(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_int, TYPE_CHECK(int*, (ptr)), descr, needed}
#define JSON_LONG(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_long, TYPE_CHECK(long*, (ptr)), descr, needed}
#define JSON_FLOAT(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_float, TYPE_CHECK(float*, (ptr)), descr, needed}
#define JSON_STRING(label, ptr, needed,  descr)	{ARRAY_SIZE((const char*[]){JSON_LABEL(label)}), (const char*[]){label}, opt_json_string, TYPE_CHECK(const char*, (ptr)), descr, needed}

extern void read_json(const char* filename, int n, const struct opt_json_s opts[n]);

#include "misc/cppwrap.h"


#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/cJSON.h"

#include "opts_json.h"

static char* readfile(const char *filename)
{
	// open file for reading
	FILE *f = NULL;
	f = fopen(filename, "r");
	
	// determine its length
	long len = 0;
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *data = NULL;
	data = (char*)malloc(len + 1);
	
	fread(data, 1, len, f);
	data[len] = '\0';
	fclose(f);

	return data;
}

static void check_json_file(cJSON *json_file)
{
	if (json_file == NULL)
	{
		const char *error_ptr = cJSON_GetErrorPtr();
		
		if (error_ptr != NULL) 
			fprintf(stderr, "Error before: %s\n", error_ptr);
		
		cJSON_Delete(json_file);
	}
}


void read_json(const char* filename, int n, const struct opt_json_s opts[n]) {

	// Import json geometry to ellipsis_e struct
	char* file = readfile(filename);

	cJSON* json_data = cJSON_Parse(file);
	check_json_file(json_data);

	for (int i = 0; i < n; i++) {

		cJSON* tmp = json_data;
		for (int j = 0; j < opts[i].n_labels - 1; j++)
			tmp = cJSON_GetObjectItemCaseSensitive(tmp, opts[i].label[j]);

		bool found = opts[i].fun(opts[i].ptr, opts[i].label[opts[i].n_labels - 1], tmp);
		
		if( opts[i].needed && !found) {

			debug_printf(DP_WARN, "Key not found in json: \"%s\"", opts[i].label[0]);
			for (int j = 1; j < opts[i].n_labels; j++)
				debug_printf(DP_WARN, "->\"%s\"", opts[i].label[j]);
			error("Key not found!");
		}
	}

	cJSON_Delete(json_data);
}


static bool cJSON_not_NULL(const cJSON* object, const char* key)
{
	return (   (NULL != cJSON_GetObjectItemCaseSensitive(object, key)) 
		&& !cJSON_IsNull(cJSON_GetObjectItemCaseSensitive(object, key)));
}

static int cJSON_get_int(const cJSON* object, const char* key)
{
	if(!cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(object, key)))
		error("cJSON: %s is not a number!\n");
	return cJSON_GetObjectItemCaseSensitive(object, key)->valueint;
}

static float cJSON_get_float(const cJSON* object, const char* key)
{
	if(!cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(object, key)))
		error("cJSON: %s is not a number!\n");
	return cJSON_GetObjectItemCaseSensitive(object, key)->valuedouble;
}

static bool cJSON_get_bool(const cJSON* object, const char* key)
{
	if(!cJSON_IsBool(cJSON_GetObjectItemCaseSensitive(object, key)))
		error("cJSON: %s is not a boolean!\n");
	return cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(object, key));
}

static const char* cJSON_get_string(const cJSON* object, const char* key)
{
	if(!cJSON_IsString(cJSON_GetObjectItemCaseSensitive(object, key)) || (NULL == cJSON_GetObjectItemCaseSensitive(object, key)->valuestring))
		error("cJSON: %s is not a string!\n");

	const char* tmp = cJSON_GetObjectItemCaseSensitive(object, key)->valuestring;
	PTR_ALLOC(char[strlen(tmp) + 1], x);
	strcpy(*x, tmp);

	return *PTR_PASS(x);
}

bool opt_json_long(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) {

		*((long*)dst) = cJSON_get_int(object, key);
		return true;
	}
	return false;
}

bool opt_json_int(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) {

		*((int*)dst) = cJSON_get_int(object, key);
		return true;
	}
	return false;
}

bool opt_json_uint(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) {

		if (0 > cJSON_get_int(object, key))
			error("%s not positive!\n");
		*((uint*)dst) = cJSON_get_int(object, key);
		return true;
	}
	return false;
}

bool opt_json_bool(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) {

		*((bool*)dst) = cJSON_get_bool(object, key);
		return true;
	}
	return false;
}

bool opt_json_float(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) { 

		*((float*)dst) = cJSON_get_float(object, key);
		return true;
	}
	return false;
}

bool opt_json_string(void* dst, const char* key, const cJSON* object)
{
	if (cJSON_not_NULL(object, key)) { 

		*((const char**)dst) = cJSON_get_string(object, key);
		return true;
	}
	return false;
}




#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "misc/cJSON.h"

#include "read_json.h"

char* readfile(const char *filename)
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


void read_json_to_struct(int N, struct ellipsis_s phantom[N], const cJSON* json_object)
{
	int i = 0;
	const cJSON *iterate = NULL;
	cJSON_ArrayForEach(iterate, json_object)
	{
		//cJSON *type = cJSON_GetObjectItemCaseSensitive(iterate, "type"); unused yet
		
		cJSON *intensity = cJSON_GetObjectItemCaseSensitive(iterate, "intensity");
		phantom[i].intensity = intensity->valuedouble;
		
		const cJSON *axis = NULL;
		axis = cJSON_GetObjectItemCaseSensitive(iterate, "axis");
		
		const cJSON *entries = NULL;
		int j = 0;
		cJSON_ArrayForEach(entries, axis)
		{
			phantom[i].axis[j] = entries->valuedouble;
			j++;
		}
		
		const cJSON *center = NULL;
		center = cJSON_GetObjectItemCaseSensitive(iterate, "center");
		
		j = 0;
		cJSON_ArrayForEach(entries, center)
		{
			phantom[i].center[j] = entries->valuedouble;
			j++;
		}
		
		cJSON *angle = cJSON_GetObjectItemCaseSensitive(iterate, "angle");
		phantom[i].angle = angle->valuedouble;
		
		//cJSON *background = cJSON_GetObjectItemCaseSensitive(iterate, "background"); unused yet

		i++;
	}
}

void check_json_file(cJSON *json_file)
{
	if (json_file == NULL)
	{
		const char *error_ptr = cJSON_GetErrorPtr();
		
		if (error_ptr != NULL) 
			fprintf(stderr, "Error before: %s\n", error_ptr);
		
		cJSON_Delete(json_file);
	}
}

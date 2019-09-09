

struct HSFP_model {
	
	float t1;
	float t2;
	float tr;
	int repetitions;
	float beta;
	complex float* pa_profile; /*Polar angle */
};
extern const struct HSFP_model hsfp_defaults;

void hsfp_simu(const struct HSFP_model* data, float* out);

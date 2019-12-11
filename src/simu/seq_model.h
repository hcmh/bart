

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


struct SimData;

struct LookLocker_model {
	
	float t1;
	float m0;
	float tr;
	float fa;
	int repetitions;
};
extern const struct LookLocker_model looklocker_defaults;

void looklocker_analytical(struct SimData* simu_data, complex float* out);


struct IRbSSFP_model {
	
	float t1;
	float t2;
	float m0;
	float tr;
	float fa;
	int repetitions;
};
extern const struct IRbSSFP_model IRbSSFP_defaults;

void IR_bSSFP_analytical(struct SimData* simu_data, complex float* out);



struct hsfp_model {
	
	float t1;
	float t2;
	float tr;
	int repetitions;
	float beta;
	complex float* pa_profile; /*Polar angle */
};

extern const struct hsfp_model hsfp_defaults;

extern void hsfp_simu(const struct hsfp_model* data, float* out);



struct LookLocker_model {
	
	float t1;
	float m0;
	float tr;
	float fa;
	int repetitions;
};

extern const struct LookLocker_model looklocker_defaults;

extern void looklocker_model(const struct LookLocker_model* data, complex float* out);



struct IRbSSFP_model {
	
	float t1;
	float t2;
	float m0;
	float tr;
	float fa;
	int repetitions;
};

extern const struct IRbSSFP_model IRbSSFP_defaults;

extern void IR_bSSFP_model(const struct IRbSSFP_model* data, complex float* out);


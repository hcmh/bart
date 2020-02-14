

struct hsfp_model {
	
	float t1;
	float t2;
	float tr;
	float beta;
};

extern const struct hsfp_model hsfp_defaults;

extern void hsfp_simu(const struct hsfp_model* data, int N, const float pa[N], complex float out[N]);



struct LookLocker_model {
	
	float t1;
	float m0;
	float tr;
	float fa;
};

extern const struct LookLocker_model looklocker_defaults;

extern void looklocker_model(const struct LookLocker_model* data, int N, complex float out[N]);



struct IRbSSFP_model {
	
	float t1;
	float t2;
	float m0;
	float tr;
	float fa;
};

extern const struct IRbSSFP_model IRbSSFP_defaults;

extern void IR_bSSFP_model(const struct IRbSSFP_model* data, int N, complex float out[N]);




struct signal_model {
	
	float m0;
	float t1;
	float t2;
	float tr;
	float fa;
	float beta;
};

extern const struct signal_model signal_hsfp_defaults;

extern void hsfp_simu(const struct signal_model* data, int N, const float pa[N], complex float out[N]);


extern const struct signal_model signal_looklocker_defaults;

extern void looklocker_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_IR_bSSFP_defaults;

extern void IR_bSSFP_model(const struct signal_model* data, int N, complex float out[N]);


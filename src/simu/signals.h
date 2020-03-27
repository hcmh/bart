
#include <complex.h>
#include <stdbool.h>

struct signal_model {
	
	float m0;
	float m0_water;
	float m0_fat;
	float t1;
	float t2;
	float t2star;
	float te;
	float tr;
	float delta_b0;
	float fa;
	float beta;
	bool ir;

};

extern const struct signal_model signal_hsfp_defaults;

extern void hsfp_simu(const struct signal_model* data, int N, const float pa[N], complex float out[N]);


extern const struct signal_model signal_looklocker_defaults;

extern void looklocker_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_IR_bSSFP_defaults;

extern void IR_bSSFP_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_multi_grad_echo_defaults;

extern void multi_grad_echo_model(const struct signal_model* data, int N, complex float out[N]);


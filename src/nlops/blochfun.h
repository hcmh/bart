
struct nlop_s;
struct noir_model_conf_s;

struct modBloch_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
};


struct modBlochFit {
	
	/*Simulation Parameter*/
	int sequence;
	float rfduration;
	float tr;
	float te;
	int averageSpokes;
	int n_slcp;
	
	/*Reconstruction Parameter*/
	float r1scaling;
	float r2scaling;
	float m0scaling;
	float fov_reduction_factor;
	int rm_no_echo;

};

extern const struct modBlochFit modBlochFit_defaults;

extern struct nlop_s* nlop_Bloch_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long input_dims[N],const complex float* input_img, const complex float* input_sp, const struct modBlochFit* fitPara, bool use_gpu);


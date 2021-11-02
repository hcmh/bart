#include "misc/cppwrap.h"

#include "misc/mri.h"


struct noir2_conf_s {

	unsigned int iter;
	_Bool rvc;
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	_Bool sms;
	_Bool sos;

	float scaling;
	_Bool undo_scaling;

	_Bool noncart;

	unsigned long enlive_flags;
	unsigned long loop_flags;

	struct nufft_conf_s* nufft_conf;

	struct opt_reg_s* regs;
	float admm_rho;

	_Bool gpu;

	int cgiter;
	float cgtol;

	int nr_init;

	_Bool real_time;
	float temp_damp;

	_Bool primal_dual;
};

extern const struct noir2_conf_s noir2_defaults;

extern void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img,
	const long img_ref_dims[N], const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens,
	const long col_ref_dims[N], const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N]);

extern void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img, const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens, const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long pat_dims[N], const complex float* pattern,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N]);

extern void noir2_rtrecon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], complex float* img,
	const long img1_dims[N], const complex float* img_ref,
	const long col_dims[N], complex float* sens, complex float* ksens,
	const long col1_dims[N], const complex float* sens_ref,
	const long ksp_dims[N], const complex float* kspace,
	const long trj_dims[N], const complex float* traj,
	const long wgh_dims[N], const complex float* weights,
	const long bas_dims[N], const complex float* basis,
	const long msk_dims[N], const complex float* mask,
	const long cim_dims[N]);

#include "misc/cppwrap.h"


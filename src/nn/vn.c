#include <assert.h>

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/rand.h"

#include "iter/proj.h"
#include <math.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter6.h"
#include "iter/batch_gen.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"

#include "nn/layers.h"
#include "nn/losses.h"
#include "nn/rbf.h"

#include "vn.h"

/*
Conventions:

Ux, Uy, Uz - dimensions, kspace dimensions
Nkx, Nky, Nkz - dimensions of convolution kernel
Nf - number of filters in convolution kernel
Nc - number of coils
Nw - resolution of field of expert
Nl - number of layers

dims = (Nx, Ny, Nz, Nc, Nb)

Input tensors:

u0:		udims = (Nx, Ny, Nz, Nb)
kspace:		kdims = (Nx, Ny, Nz, Nc, Nb)
coils:		cdims = (Nx, Ny, Nz, Nc, Nb)
mask:		mdims = (Nx, Ny, Nz, 1, 1)

lambda: 	ldims = (1,  Nl)
kernel: 	kerdims = (Nf, Nkx, Nky, Nkz, Nl)
weights: 	wdims = (Nf, Nw , Nl)

Output tensor:

ul:		udims = (Nx, Ny, Nz, Nb)
*/


const struct vn_s vn_default = {

	.Nl = 10,

	.Nf = 24,

	.Kx = 11,
	.Ky = 11,
	.Kz = 1,

	.Px = 11,
	.Py = 11,
	.Pz = 0,

	.Nw = 31,
	.Imax = 1.,
	.Imin = -1.,

	.conv_w = NULL,
	.rbf_w = NULL,
	.lambda_w = NULL,

	.conv_w_cpu = NULL,
	.rbf_w_cpu = NULL,
	.lambda_w_cpu = NULL,

	.lambda_init = .2,
	.init_scale_mu = 0.04,

	.share_mask = true,
};


/**
 * Returns operator computing the update due to the regularizer
 * 1/Nf \sum_{i=0}^{N_k} K_i^T \Phi'_i(K_i u)
 *
 * Input tensors: 	(u, conv_w, rbf_w)
 *
 * u: 		udims:	(Ux, Uy, Uz, 1, Nb)
 * conv_w:	kerdims:(Nf, Kx, Ky, Kz, 1)
 * rbf_w:	wdims:	(Nf, Nw, 1)
 *
 * Output tensors:
 *
 * Ru:	 	udims:	(Ux, Uy, Uz, 1, Nb)
 */
static const struct nlop_s* nlop_ru_create(struct vn_s* vn, const long udims[5])
{
	//Padding
	long pad_up[5] = {0, vn->Px, vn->Py, vn->Pz, 0};
	long pad_down[5] = {0, -vn->Px, -vn->Py, -vn->Pz, 0};
	long ker_size[3] = {vn->Kx, vn->Ky, vn->Kz};

	long Ux = udims[0];
	long Uy = udims[1];
	long Uz = udims[2];
	long Nb = udims[4];

	//working dims
	long udimsw[5] = {1, Ux, Uy, Uz, Nb};
	long zdimsw[5] = {vn->Nf, Ux + 2 * vn->Px, Uy + 2 * vn->Py, Uz + 2 * vn->Pz, Nb};
	long rbfdims[3] = {vn->Nf, (Ux + 2 * vn->Px) * (Uy + 2 * vn->Py) * (Uz + 2 * vn->Pz) * Nb, vn->Nw};

	//operator dims
	long kerdims[5] = {vn->Nf, vn->Kx, vn->Ky, vn->Kz, 1};
	long wdims[3] = {vn->Nf, vn->Nw, 1};

	const struct nlop_s* result = nlop_from_linop_F(linop_reshape_create(5, udimsw, 5, udims)); // in: u
	//result = nlop_chain2_FF(result, 0, padu, 0); // in: u
	result = append_padding_layer(result, 0, 5, pad_up, pad_up, PAD_SYMMETRIC);
	result = append_convcorr_layer(result, 0, vn->Nf, ker_size, false, PAD_SAME, true, NULL, NULL); // in: u, conv_w

	const struct nlop_s* rbf = nlop_activation_rbf_create(rbfdims, vn->Imax, vn->Imin);
	rbf = nlop_reshape_in_F(rbf, 0, 5, zdimsw);
	rbf = nlop_reshape_out_F(rbf, 0, 5, zdimsw);
	result = nlop_chain2_FF(result, 0, rbf, 0); //in: rbf_w, in, conv_w

	result = append_transposed_convcorr_layer(result, 0, 1, ker_size, false, true, PAD_SAME, true, NULL, NULL); //in: rbf_w, u, conv_w, conv_w
	//result = nlop_chain2_FF(result, 0, padd, 0); //in: rbf_w, u, conv_w, conv_w
	result = append_padding_layer(result, 0, 5, pad_down, pad_down, PAD_VALID);
	result = nlop_dup_F(result, 2, 3); //in: rbf_w, u, conv_w

	result = nlop_reshape_out_F(result, 0, 5, udims); //in: rbf_w, u, conv_w
	result = nlop_reshape_in_F(result, 2, 5, kerdims); //in: rbf_w, u, conv_w
	result = nlop_reshape_in_F(result, 0, 3, wdims); //in: rbf_w, u, conv_w

	//VN implementation: u_k = (real(up) * real(k) + imag(up) * imag(k))
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(5, kerdims)), 0, result, 2); //in: rbf_w, u, conv_w
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_scale_create(5, udims, 1. / vn->Nf)), 0); //in: rbf_w, u, conv_w

	int perm [] = {1, 2, 0};
	result = nlop_permute_inputs_F(result, 3, perm); //in: u, conv_w, rbf_w

	debug_printf(DP_DEBUG3, "RU created: ");
	nlop_debug(DP_DEBUG3, result);

	return  result;
}

/**
 * Returns operator computing update for data fidelity
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 * @param udims (Ux, Uy, Uz, 1, Nb)
 *
 * Input tensors:
 * u:		udims: 	(Ux, Uy, Uz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda:	dims:	(1)
 *
 * Output tensors:
 * Du:		udims: 	(Ux, Uy, Uz, 1,  Nb)
 */
static const struct nlop_s* nlop_du_create(struct vn_s* vn, const long dims[5], const long udims[5])
{
	const struct nlop_s* result = nlop_gradient_step_scaled_create(dims, vn->share_mask);

	if (!md_check_equal_dims(5, udims, nlop_generic_codomain(result, 0)->dims, ~0))
		result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_resize_center_create(5, udims, nlop_generic_codomain(result, 0)->dims)), 0);

	if (!md_check_equal_dims(5, udims, nlop_generic_domain(result, 0)->dims, ~0)) {

		result = nlop_chain2_FF(nlop_from_linop_F(linop_resize_center_create(5, nlop_generic_domain(result, 0)->dims, udims)), 0, result, 0);
		result = nlop_permute_inputs_F(result, 5, MAKE_ARRAY(4, 0, 1, 2, 3));
	}

	debug_printf(DP_DEBUG3, "DU created: ");
	nlop_debug(DP_DEBUG3, result);

	return result;
}


/**
 * Returns cell of variational network
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 *
 * Input tensors:
 * u(t):	udims:	(Ux, Uy, Uz, 1,  Nb)
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * conv_w(t):	kerdims:(Nf, Kx, Ky, Kz, 1)
 * rbf_w(t):	wdims:	(Nf, Nw, 1)
 * l(t):	ldims:	(1,  1)
 *
 * Output tensors:
 * u(t+1):	udims:	(Ux, Uy, Uz, 1, Nb)
 */
static const struct nlop_s* get_vn_cell(struct vn_s* vn, const long dims[5], const long udims[5])
{
	const struct nlop_s* ru = nlop_ru_create(vn, udims); //in: u(t), conv_w, rbf_w

	const struct nlop_s* du = nlop_du_create(vn, dims, udims); //in: u(t), kspace, coil, mask, lambda
	du = nlop_reshape_in_F(du, 4, 2, MD_SINGLETON_DIMS(2)); // in: u(t), kspace, coil, mask, lambda

	const struct nlop_s* result = nlop_combine_FF(du, ru); // in: u(t), kspace, coil, mask, lambda, u(t), conv_w, rbf_w; out: du, ru
	result = nlop_dup_F(result, 0, 5); // in: u(t), lambda, kspace, coil, mask, conv_w, rbf_w; out: du, ru

	result = nlop_combine_FF(nlop_zaxpbz_create(5, udims, 1., 1.), result); // in: du, ru, u(t), kspace, coil, mask, lambda, conv_w, rbf_w; out: du+ru, du, ru
	result = nlop_link_F(result, 1, 0);
	result = nlop_link_F(result, 1, 0); // in: u(t), lambda, kspace, coil, mask, conv_w, rbf_w; out: du+ru

	result = nlop_combine_FF(nlop_zaxpbz_create(5, udims, 1., -1.), result); // in: u(t), (du+ru), u(t), kspace, coil, mask, lambda, conv_w, rbf_w; out: u(t+1)=u(t)-(du+ru), du+ru
	result = nlop_dup_F(result , 0, 2); // in: u(t), (du+ru), lambda, kspace, coil, mask, conv_w, rbf_w; out: u(t+1), (du+ru)
	result = nlop_link_F(result, 1, 1); // in: u(t), kspace, coil, mask, lambda, conv_w, rbf_w; out: u(t+1)

	result = nlop_permute_inputs_F(result, 7, MAKE_ARRAY(0, 1, 2, 3, 5, 6, 4)); // in: u(t), kspace, coil, mask, conv_w, rbf_w, lambda; out: u(t+1)

	debug_printf(DP_DEBUG3, "VN CELL created: ");
	nlop_debug(DP_DEBUG3, result);

	return result;
}

/**
 * Returns variational network
 *
 * @param vn structure describing the variational network
 * @param dims (Nx, Ny, Nz, Nc, Nb)
 *
 * Input tensors:
 * kspace:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * coils:	kdims:	(Nx, Ny, Nz, Nc, Nb)
 * mask:	mdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * conv_w[0:1]:	kerdims:(Nf, Kx, Ky, Kz, Nl)
 * rbf_w[0:1]:	wdims:	(Nf, Nw , Nl)
 * l[0:1]:	ldims:	(1,  Nl)

 *
 * Output tensors:
 * u(l):	udims:	(Ux, Uy, Uz, 1,  Ub)
 */
static const struct nlop_s* get_variational_network(struct vn_s* vn, const long dims[5], const long udims[5])
{
	const struct nlop_s* result = get_vn_cell(vn, dims, udims);

	for (int l = 1; l < vn->Nl; l++) {

		const struct nlop_s* tmp = get_vn_cell(vn, dims, udims);

		result = nlop_chain2_FF(result, 0, tmp, 0); //in: kspace, coil, mask, conv_w[l], rbf_w[l], lambda[l], u(0), kspace, coil, mask, conv_w[0:l-1], rbf_w[0:l-1], lambda[0:l-1]

		result = nlop_dup_F(result, 0, 7);
		result = nlop_dup_F(result, 1, 7);
		result = nlop_dup_F(result, 2, 7); //in: kspace, coil, mask, conv_w[l], rbf_w[l], lambda[l], u(0), conv_w[0:l-1], rbf_w[0:l-1], lambda[0:l-1]

		result = nlop_destack_F(result, 7, 3, 4);
		result = nlop_destack_F(result, 7, 4, 2);
		result = nlop_destack_F(result, 7, 5, 1); //in: kspace, coil, mask, conv_w[0:l], rbf_w[0:l], lambda[0:l], u(0)

		result = nlop_permute_inputs_F(result, 7, MAKE_ARRAY(6, 0, 1, 2, 3, 4, 5));
	}

	auto adjoint_op = nlop_mri_adjoint_create(dims, vn->share_mask);

	if (!md_check_equal_dims(5, nlop_generic_codomain(adjoint_op, 0)->dims, nlop_generic_domain(result, 0)->dims, ~0))
		adjoint_op = nlop_chain2_FF(adjoint_op, 0, nlop_from_linop_F(linop_resize_center_create(5, nlop_generic_domain(result, 0)->dims, nlop_generic_codomain(adjoint_op, 0)->dims)), 0);

	result = nlop_chain2_FF(adjoint_op, 0, result, 0); //in: kspace, coil, mask, conv_w[0:l], rbf_w[0:l], lambda[0:l], kspace, coil, mask

	result = nlop_dup_F(result, 0, 6);
	result = nlop_dup_F(result, 1, 6);
	result = nlop_dup_F(result, 2, 6); //in: kspace, coil, mask, conv_w[0:l], rbf_w[0:l], lambda[0:l]

	reset_fft_op();

	debug_printf(DP_DEBUG3, "VN created: ");
	nlop_debug(DP_INFO, result);

	return result;
}



/**
 * Creates Variational Network and applies it
 *
 * @param vn structure describing the variational network
 * @param udims (Ux, Uy, Uz, 1, Nb) - image dims
 * @param out pointer to output array
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param mdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of mask
 * @param coils pointer to mask data
 */
void apply_variational_network( struct vn_s* vn,
 				const long udims[5], complex float* out,
				const long kdims[5], const complex float* kspace, const complex float* coil, const long mdims[5], const complex float* mask)
{

	vn->share_mask = mdims[4] == 1;

	auto network = get_variational_network(vn, kdims, udims);

	complex float* out_tmp = md_alloc_sameplace(5, udims, CFL_SIZE, vn->conv_w);
	complex float* kspace_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, vn->conv_w);
	complex float* coil_tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, vn->conv_w);
	complex float* mask_tmp = md_alloc_sameplace(5, mdims, CFL_SIZE, vn->conv_w);

	md_copy(5, kdims, kspace_tmp, kspace, CFL_SIZE);
	md_copy(5, kdims, coil_tmp, coil, CFL_SIZE);
	md_copy(5, mdims, mask_tmp, mask, CFL_SIZE);

	void* refs[] = {(void*)out_tmp, (void*)kspace_tmp, (void*)coil_tmp, (void*)mask_tmp, (void*)vn->conv_w, (void*)vn->rbf_w, (void*)vn->lambda_w};

	nlop_generic_apply_select_derivative_unchecked(network, 7, refs, 0, 0);

	md_copy(5, udims, out, out_tmp, CFL_SIZE);

	nlop_free(network);
	md_free(out_tmp);
	md_free(kspace_tmp);
	md_free(coil_tmp);
	md_free(mask_tmp);
}


/**
 * Creates Variational Network and applies it batchwise
 *
 * @param vn structure describing the variational network
 * @param udims (Ux, Uy, Uz, 1, Nb) - image dims
 * @param out pointer to output array
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param mdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of mask
 * @param coils pointer to mask data
 * @param Nb batch size for applying vn
 */
void apply_variational_network_batchwise(	struct vn_s* vn,
						const long udims[5], complex float * out,
						const long kdims[5], const complex float* kspace, const complex float* coil, const long mdims[5], const complex float* mask,
						long Nb)
{
	long Nt = kdims[4];
	if (Nt <= Nb) {

			apply_variational_network(vn, udims, out, kdims, kspace, coil, mdims, mask);
	} else {

		long kdims1[5];
		long kdims2[5];
		long udims1[5];
		long udims2[5];
		long mdims1[5];
		long mdims2[5];

		md_copy_dims(5, kdims1, kdims);
		md_copy_dims(5, kdims2, kdims);
		md_copy_dims(5, udims1, udims);
		md_copy_dims(5, udims2, udims);
		md_copy_dims(5, mdims1, mdims);
		md_copy_dims(5, mdims2, mdims);

		kdims1[4] = Nb;
		kdims2[4] = Nt - Nb;
		udims1[4] = Nb;
		udims2[4] = Nt - Nb;
		mdims1[4] = MIN(mdims1[4], Nb);
		mdims1[4] = MIN(mdims2[4], Nt - Nb);

		long offset_k = md_calc_size(4, kdims) * Nb;
		long offset_u = md_calc_size(4, udims) * Nb;
		long offset_m = md_calc_size(4, udims) * Nb * (mdims[4] > 1);

		apply_variational_network_batchwise(vn, udims1, out, kdims1, kspace, coil, mdims1, mask, Nb);
		apply_variational_network_batchwise(vn, udims2, out + offset_u, kdims2, kspace + offset_k, coil + offset_k, mdims2, mask + offset_m, Nb);
	}
}



/**
 * Trains Variational Network
 *
 * @param vn structure describing the variational network
 * @param train_conf structure holding training parameters of iPALM algorithm
 * @param udims (Ux, Uy, Uz, 1, Nb) - image dims of reference data
 * @param ref pointer to reference
 * @param kdims (Nx, Ny, Nz, Nc, Nb) - dims of kspace and coils
 * @param kspace pointer to kspace data
 * @param coils pointer to coil data
 * @param mdims (Nx, Ny, Nz, 1, 1 / Nb) - dims of mask
 * @param coils pointer to mask data
 * @param Nb batch size for training
 * @param random_order draw random datasets to composite batch
 */
void train_nn_varnet(	struct vn_s* vn, iter6_conf* train_conf,
			const long udims[5], const _Complex float* ref,
			const long kdims[5], const _Complex float* kspace, const _Complex float* coil,
			const long mdims[5], const _Complex float* mask,
			long Nb, bool random_order, const char* history_filename, const char** valid_files)
{
	long Nt = kdims[4]; // number datasets

	long nkdims[5];
	long nudims[5];

	md_copy_dims(5, nkdims, kdims);
	md_copy_dims(5, nudims, udims);

	nkdims[4] = Nb;
	nudims[4] = Nb;

	vn->share_mask = mdims[4] == 1;

	auto network = get_variational_network(vn, nkdims, nudims);

	//append loss = 1/(2N) sum_i^N || |x_i| - |y_i| ||^2 with |x_i| = sqrt(Re[x_i]^2 + Im[x_i]^2 + epsilon)
	const struct nlop_s* loss = nlop_mse_create(5, nudims, MD_BIT(4));
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, nudims, 1.e-12), 0, loss, 0);
	loss = nlop_chain2_FF(nlop_smo_abs_create(5, nudims, 1.e-12), 0, loss, 0);

	network = nlop_chain2_FF(network, 0, loss, 0);

	debug_printf(DP_DEBUG3, "train op created: ");
	nlop_debug(DP_DEBUG3, network);

	//create batch generator
	const complex float* train_data[] = {ref, kspace, coil, mask};
	const long* train_dims[] = {	nlop_generic_domain(network, 0)->dims,
					nlop_generic_domain(network, 1)->dims,
					nlop_generic_domain(network, 2)->dims,
					nlop_generic_domain(network, 3)->dims};

	struct iter6_iPALM_conf* iPALM_conf = CAST_DOWN(iter6_iPALM_conf, train_conf);

	auto batch_generator = (random_order ? batch_gen_rand_create : batch_gen_linear_create)(4, 5, train_dims, train_data, Nt, iPALM_conf->epoch_start);

	//setup for iter algorithm
	float* data[7];
	enum IN_TYPE in_type[7];
	const struct operator_p_s* projections[7];

	for (int i = 0; i < 4; i++) {

		data[i] = NULL;
		in_type[i] = IN_BATCH_GENERATOR;
		projections[i] = NULL;
	}

	data[4] = (float*)vn->conv_w;
	data[5] = (float*)vn->rbf_w;
	data[6] = (float*)vn->lambda_w;

	projections[4] = operator_project_mean_free_sphere_create(nlop_generic_domain(network, 4)->N, nlop_generic_domain(network, 4)->dims, MD_BIT(0) | MD_BIT(4), false);
	projections[5] = NULL;
	projections[6] = operator_project_pos_real_create(nlop_generic_domain(network, 6)->N, nlop_generic_domain(network, 6)->dims);

	in_type[4] = IN_OPTIMIZE;
	in_type[5] = IN_OPTIMIZE;
	in_type[6] = IN_OPTIMIZE;

	char* names[] = {NULL, NULL, NULL, NULL, "conv_w", "rbf_w", "lambda_w"};
	iPALM_conf->save_name = names;

	const struct nlop_s* valid_loss = NULL;
	if (NULL != valid_files) {

		long kdims[5];
		long cdims[5];
		long udims[5];
		long mdims[5];

		complex float* val_kspace = load_cfl(valid_files[0], 5, kdims);
		complex float* val_coil = load_cfl(valid_files[1], 5, cdims);
		complex float* val_mask = load_cfl(valid_files[2], 5, mdims);
		complex float* val_ref = load_cfl(valid_files[3], 5, udims);

		complex float* scaling = md_alloc(1, MAKE_ARRAY(kdims[4]), CFL_SIZE);
		complex float* u0 = md_alloc(5, udims, CFL_SIZE);
		compute_zero_filled(udims, u0, kdims, val_kspace, val_coil, mdims, val_mask);
		compute_scale(udims, scaling, u0);
		md_free(u0);

		normalize_max(udims, scaling, val_ref, val_ref);
		normalize_max(kdims, scaling, val_kspace, val_kspace);

		valid_loss = get_variational_network(vn, kdims, udims);

		const struct nlop_s* loss = nlop_mse_create(5, udims, MD_BIT(4));
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);
		loss = nlop_chain2_FF(nlop_smo_abs_create(5, udims, 1.e-12), 0, loss, 0);

		valid_loss = nlop_chain2_FF(valid_loss, 0, loss, 0);

		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, udims, true, val_ref);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, kdims, true, val_kspace);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, cdims, true, val_coil);
		valid_loss = nlop_set_input_const_F(valid_loss, 0, 5, mdims, true, val_mask);

		unmap_cfl(5, udims, val_ref);
		unmap_cfl(5, kdims, val_kspace);
		unmap_cfl(5, cdims, val_coil);
		unmap_cfl(5, mdims, val_mask);
	}

	//auto monitor = create_iter6_monitor_progressbar(Nt / Nb, true);
	auto conf = CAST_DOWN(iter6_iPALM_conf, train_conf);
	auto monitor = create_iter6_monitor_progressbar_validloss(conf->epochs, Nt / Nb, false, 7, in_type, valid_loss, true);
	iter6_iPALM(train_conf, network, 7, in_type, projections, data, 1, MAKE_ARRAY((enum OUT_TYPE)OUT_OPTIMIZE), 0, Nt / Nb, batch_generator, monitor);
	if (NULL != history_filename)
		iter6_monitor_dump_record(monitor, history_filename);

	nlop_free(network);
	nlop_free(batch_generator);
}

/**
 * Move vn weights to gpu / cpu
 */
void vn_move_gpucpu(struct vn_s* vn, bool gpu) {
	#ifdef USE_CUDA

	if ((NULL == vn->conv_w) || (NULL == vn->rbf_w) || (NULL == vn->lambda_w))
		error("vn uninitialized\n");

	long conv_size = vn->Nl * vn->Kx * vn->Ky * vn->Kz * vn->Nf;
	long rbf_size = vn->Nl * vn->Nf * vn->Nw;
	long lambda_size = vn->Nl;

	complex float* conv_w = (gpu ? md_alloc_gpu: md_alloc)(1, &conv_size, CFL_SIZE);
	complex float* rbf_w = (gpu ? md_alloc_gpu: md_alloc)(1, &rbf_size, CFL_SIZE);
	complex float* lambda_w = (gpu ? md_alloc_gpu: md_alloc)(1, &lambda_size, CFL_SIZE);

	md_copy(1, &conv_size, conv_w, vn->conv_w, CFL_SIZE);
	md_copy(1, &rbf_size, rbf_w, vn->rbf_w, CFL_SIZE);
	md_copy(1, &lambda_size, lambda_w, vn->lambda_w, CFL_SIZE);

	md_free(vn->conv_w);
	vn->conv_w = conv_w;
	md_free(vn->rbf_w);
	vn->rbf_w = rbf_w;
	md_free(vn->lambda_w);
	vn->lambda_w = lambda_w;
	#else
	UNUSED(vn);
	if (gpu)
		error("compiled without cuda support\n");
	#endif
}

void initialize_varnet(struct vn_s* vn)
{
	long conv_size = vn->Nl * vn->Kx * vn->Ky * vn->Kz * vn->Nf;
	long rbf_size = vn->Nl * vn->Nf * vn->Nw;
	long lambda_size = vn->Nl;

	if (NULL == vn->conv_w)
		vn->conv_w = md_alloc(1, &conv_size, CFL_SIZE);
	if (NULL == vn->rbf_w)
		vn->rbf_w = md_alloc(1, &rbf_size, CFL_SIZE);
	if (NULL == vn->lambda_w)
		vn->lambda_w = md_alloc(1, &lambda_size, CFL_SIZE);

	md_zfill(1, &lambda_size, vn->lambda_w, vn->lambda_init);

	complex float mu[vn->Nw];
	for (int i = 0; i < vn->Nw; i++)
		mu[i] = vn->init_scale_mu * (vn->Imin + (i)*(vn->Imax - vn->Imin) / ((float)vn->Nw - 1.));
	long wdims[3] = {vn->Nf, vn->Nw, vn->Nl};
	md_copy2(3, wdims, MD_STRIDES(3, wdims, CFL_SIZE), vn->rbf_w, MAKE_ARRAY(0l, 8l, 0l) , mu, CFL_SIZE);

	long kdims[5] = {vn->Nf, vn->Kx, vn->Ky, vn->Kz, vn->Nl};
	complex float* tmp = md_alloc_sameplace(5, kdims, CFL_SIZE, vn->conv_w);
	md_gaussian_rand(5, kdims, tmp);
	md_zsmax(5, kdims, tmp, tmp, I);
	md_gaussian_rand(5, kdims, vn->conv_w);
	md_zadd(5, kdims, vn->conv_w, vn->conv_w, tmp);
	md_free(tmp);
	md_zsmul(5, kdims, vn->conv_w, vn->conv_w, 1. / sqrtf((float)vn->Kx * vn->Ky * vn->Kz));
}

void save_varnet(struct vn_s* vn, const char* filename_conv_w, const char* filename_rbf_w, const char* filename_lambda)
{
	long dims_conv_w[5] = {vn->Nf, vn->Kx, vn->Ky, vn->Kz, vn->Nl};
	long dims_rbf_w[3] = {vn->Nf, vn->Nw, vn->Nl};
	long dims_lambda_w[2] = {1, vn->Nl};

	dump_cfl(filename_conv_w, 5, dims_conv_w, vn->conv_w);
	dump_cfl(filename_rbf_w, 3, dims_rbf_w, vn->rbf_w);
	dump_cfl(filename_lambda, 2, dims_lambda_w, vn->lambda_w);
}

/**
 * Compute zero filled reconstruction
 */
void compute_zero_filled(const long udims[5], complex float * out, const long kdims[5], const complex float* kspace, const complex float* coil, long mdims[5], const complex float* mask)
{
	auto mri_adjoint = nlop_mri_adjoint_create(kdims, mdims[4] == 1);

	complex float* tmp = md_alloc_sameplace(5, nlop_generic_codomain(mri_adjoint, 0)->dims, CFL_SIZE, out);
	nlop_generic_apply_unchecked(mri_adjoint, 4, MAKE_ARRAY((void*)tmp, (void*)kspace, (void*)coil, (void*)mask));

	md_resize_center(5, udims, out, nlop_generic_codomain(mri_adjoint, 0)->dims, tmp, CFL_SIZE);
	nlop_free(mri_adjoint);
	md_free(tmp);
}

/**
 * Compute maximum of u0 per batch
 */
void compute_scale(const long dims[5], complex float* scaling, const complex float * u0)
{
	long udimsw[5]; // (Nx, Ny, Nz, 1, Nb)
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, ~MD_BIT(3), udimsw, dims);
	md_select_dims(5, MD_BIT(4), sdimsw, dims);

	complex float* u0_abs = md_alloc_sameplace(5, udimsw, CFL_SIZE, u0);
	md_zabs(5, udimsw, u0_abs, u0);

	md_clear(5, sdimsw, scaling, CFL_SIZE);
	md_zmax2(5, udimsw, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling, MD_STRIDES(5, udimsw, CFL_SIZE), u0_abs);

	md_free(u0_abs);
}


/**
 * Batchwise divide by scaling
 */
void normalize_max(const long dims[5], const complex float* scaling, complex float* out, const complex float* in)
{
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, MD_BIT(4), sdimsw, dims);
	md_zdiv2(5, dims, MD_STRIDES(5, dims, CFL_SIZE), out, MD_STRIDES(5, dims, CFL_SIZE), in, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling);
}

/**
 * Batchwise multiply by scaling
 */
void renormalize_max(const long dims[5], const complex float* scaling, complex float* out, const complex float* in)
{
	long sdimsw[5]; // (1,  1,  1,  1, Nb)
	md_select_dims(5, MD_BIT(4), sdimsw, dims);
	md_zmul2(5, dims, MD_STRIDES(5, dims, CFL_SIZE), out, MD_STRIDES(5, dims, CFL_SIZE), in, MD_STRIDES(5, sdimsw, CFL_SIZE), scaling);
}

/**
 * Batchwise multiply by scaling
 */

void unmask_zeros(long mdims[5], complex float* mask, long kdims[5], const complex float* kspace)
{
	if (1 != mdims[4])
		error("not supported for batchwise mask");

	complex float* tmp = md_alloc(5, kdims, CFL_SIZE);
	md_clear(5, kdims, tmp, CFL_SIZE);
	md_zcmp(5, kdims, tmp, tmp, kspace);

	complex float* tmp_mask = md_alloc(5, mdims, CFL_SIZE);
	md_clear(5, mdims, tmp_mask, CFL_SIZE);
	md_zadd2(5, kdims, MD_STRIDES(5, mdims, CFL_SIZE), tmp_mask, MD_STRIDES(5, mdims, CFL_SIZE), tmp_mask, MD_STRIDES(5, kdims, CFL_SIZE), tmp);
	md_free(tmp);

	md_zsgreatequal(5, mdims, tmp_mask, tmp_mask, kdims[3] * kdims[4]); //kspace is 0 for all batches and all coils

	md_zadd(5, mdims, tmp_mask, tmp_mask, mask);
	md_zsgreatequal(5, mdims, mask, tmp_mask, 1.);
	md_free(tmp_mask);

}
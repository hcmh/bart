#include "iter/italgos.h"
#include "linops/someops.h"
#include "nn/activation.h"

enum UNET_DOWNSAMPLING_METHODE {UNET_DS_FFT, UNET_DS_MPOOL};
enum UNET_UPSAMPLING_COMBINE {UNET_COMBINE_SUM, UNET_COMBINE_CONV};

struct unet_s {

	long convolution_kernel[3];

	long channels; //channels on highest level
	float channel_factor; //number channels on lower level
	float reduce_factor; //reduce resolution of lower level
	long number_levels;
	long number_output_channels;
	long number_layers_per_level;

	_Bool real_constraint_input;
	_Bool real_constraint_output;
	_Bool real_constraint_weights;

	_Bool use_batchnormalization;
	_Bool use_bias;
	_Bool use_transposed_convolution;

	enum ACTIVATION activation;
	enum ACTIVATION activation_output;
	enum ACTIVATION activation_last_layer;

	enum PADDING padding;

	enum UNET_DOWNSAMPLING_METHODE ds_methode;
	enum UNET_UPSAMPLING_COMBINE upsampling_combine;

	_Bool reinsert_input;
	_Bool residual;

	const struct nlop_s** deflatten_conv;
	const struct nlop_s** deflatten_bias;
	const struct nlop_s** deflatten_bn;

	_Complex float* weights;
	long size_weights;

	long size_weights_conv;
	long size_weights_bias;
	long size_weights_bn;
};

extern const struct unet_s unet_default_reco;

extern const struct nlop_s* nn_unet_create(struct unet_s* unet, long dims[5]);
extern void nn_unet_initialize(struct unet_s* unet, long dims[5]);

extern int nn_unet_get_number_in_weights(struct unet_s* unet);
extern int nn_unet_get_number_out_weights(struct unet_s* unet);

enum IN_TYPE;
enum OUT_TYPE;

extern void nn_unet_get_in_weights_pointer(struct unet_s* unet, int N, _Complex float* args[N]);
extern void nn_unet_get_in_types(struct unet_s* unet, int N, enum IN_TYPE in_types[N]);
extern void nn_unet_get_out_types(struct unet_s* unet, int N, enum OUT_TYPE out_types[N]);

extern long nn_unet_get_weights_size(struct unet_s* unet);
extern void nn_unet_load_weights(struct unet_s* unet, long size, _Complex float* in);
extern void nn_unet_store_weights(const struct unet_s* unet, long size, _Complex float* out);

extern void nn_unet_move_cpugpu(struct unet_s* unet, _Bool gpu);

extern void nn_unet_free_weights(struct unet_s* unet);
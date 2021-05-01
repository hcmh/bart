#include <complex.h>
#include <stdio.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tensor.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "tf_wrapper.h"

static void free_buffer(void* data, size_t length) { free((void*)data); }
static void deallocator(void* ptr, size_t len, void* arg) { }; //free((void*)ptr);

static int product(int n, int64_t ar[n])
{
	
    int64_t result = 1;
    for (int i = 0; i < n; i++)
		result = result * ar[i];
		
    return result;
}

static void read_gpu_config(char* path, uint8_t* config)
{
	FILE *gpu_config_file;
	gpu_config_file = fopen(path, "r");
	
	if(NULL==gpu_config_file)
		printf("gpu config file doesn't exist");

	fseek (gpu_config_file, 0, SEEK_END);
 	long lSize = ftell (gpu_config_file);
	char* buffer = (char*) malloc (sizeof(char)*lSize);
	fseek(gpu_config_file, 0, SEEK_SET);
	fread(buffer, 1, lSize, gpu_config_file);

	char tmp[10]={0};
	
	int j=0;
	int i=0;
	int u;
	
	int count = 0;
	while( buffer[i] !='\0')
	{

		if(buffer[i]!='\t')
		{
			tmp[j] = buffer[i];
			j++;
		}
		else
		{
			j=0;
			config[count] = (uint8_t)strtoul(tmp, NULL, 16);
			memset(tmp, 0, 10);
			count++;
		}
		
		i++;
	}
	
	fclose(gpu_config_file);
	
}

// function to read network/graph definition from binary protobuf file
static TF_Buffer* read_graph(const char* file)
{
	FILE* f = fopen(file, "rb");
	
	if (f == NULL)
		error("Graph file not found!\n");

	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);  // same as rewind(f);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	return buf;
}

// function to load graph
static void load_graph(TF_Buffer* graph_def, TF_Graph* graph, TF_Status* status)
{
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_def, opts, status);
	TF_DeleteImportGraphDefOptions(opts);

	if (TF_GetCode(status) != TF_OK)
		error("Loading graph failed: %s\n", TF_Message(status));

	debug_printf(DP_DEBUG3, "Graph loaded!\n");
}

// function to create session
static TF_Session* create_session(TF_Graph* graph, TF_Status* status, uint8_t* config, bool test)
{
	
	TF_SessionOptions* opt = TF_NewSessionOptions();
	
	TF_SetConfig(opt, (void*)config, strlen(config), status);
	
	if (TF_GetCode(status) != TF_OK)
		debug_printf(DP_DEBUG3, "GPU selection failed\n");
		
	TF_Session* sess = TF_NewSession(graph, opt, status);
	
	TF_DeleteSessionOptions(opt);
	if (TF_GetCode(status) != TF_OK)
		error("Unable to create session %s\n", TF_Message(status));

	debug_printf(DP_DEBUG3, "Session created.\n");

	if(test){
		const TF_Operation* init_op = TF_GraphOperationByName(graph, "init");
		const TF_Operation* const* targets_ptr = &init_op;
		TF_SessionRun(sess,
				/* RunOptions */ NULL,
				/* Input tensors */ NULL, NULL, 0,
				/* Output tensors */ NULL, NULL, 0,
				/* Target operations */ targets_ptr, 1,
				/* RunMetadata */ NULL,
				/* Output status */ status);
		if (TF_GetCode(status) != TF_OK)
			error("Unable to run init_op: %s\n", TF_Message(status));


		debug_printf(DP_DEBUG3, "Session initialized.\n");
	}

	TF_DeviceList *devices= TF_SessionListDevices(sess, status);

	int nr_gpus = TF_DeviceListCount(devices);
	debug_printf(DP_DEBUG3, "====num of devices: %d  ====\n", nr_gpus);
	for (int i=0; i<nr_gpus; i++)
	{
		char *info = TF_DeviceListName(devices, i, status);
		char *type = TF_DeviceListType(devices, i, status);
		debug_printf(DP_DEBUG3, "DEVICE %d: %s |", i, type);
		debug_printf(DP_DEBUG3, info);
		debug_printf(DP_DEBUG3, "\n");
	}
	
	return sess;
}


// function to restore trained weights
static void restore_sess(TF_Graph* graph, TF_Status *status, TF_Session *sess, const char* ckpt_path)
{
	TF_Operation* checkpoint_op = TF_GraphOperationByName(graph, "save/Const");
	const TF_Operation* restore_op = TF_GraphOperationByName(graph, "save/restore_all");

	size_t checkpoint_path_str_len = strlen(ckpt_path);
	size_t encoded_size = TF_StringEncodedSize(checkpoint_path_str_len);

	size_t total_size = sizeof(int64_t) + encoded_size;
	char* input_encoded = (char*)malloc(total_size);
	memset(input_encoded, 0, total_size);
	TF_StringEncode(ckpt_path, checkpoint_path_str_len, input_encoded + sizeof(int64_t), encoded_size, status);

	if (TF_GetCode(status) != TF_OK)
		error("Something wrong with encoding: %s", TF_Message(status));

	TF_Tensor* path_tensor = TF_NewTensor(TF_STRING, NULL, 0, input_encoded, total_size, &deallocator, 0);

	TF_Output run_path;
	run_path.oper = checkpoint_op;
	run_path.index = 0;

	TF_SessionRun(sess,
					/* RunOptions */ NULL,
					/* Input tensors */ &run_path, &path_tensor, 1,
					/* Output tensors */ NULL, NULL, 0,
					/* Target operations */ &restore_op, 1,
					/* RunMetadata */ NULL,
					/* Output status */ status);

	TF_DeleteTensor(path_tensor);

	if (TF_GetCode(status) != TF_OK)
		error("ERROR: Unable to run restore_op: %s\n", TF_Message(status));

	debug_printf(DP_DEBUG3, "Session restored.\n");

}


static TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const int64_t* dims, size_t num_dims, size_t len)
{
	if (NULL == dims)
		return NULL;

	return TF_AllocateTensor(data_type, dims, (int)(num_dims), len);
}

static TF_Tensor* CreateTensor(TF_DataType data_type,
			const int64_t* dims, size_t num_dims,
			const void* data, size_t len)
{
	void* tensor = CreateEmptyTensor(data_type, dims, num_dims, len);
	if (NULL == tensor)
		return NULL;

	void* tensor_data = TF_TensorData(tensor);
	tensor_data = (void *)data;
	if (tensor_data == NULL) {

		TF_DeleteTensor(tensor);
		return NULL;
	}

	len = MIN(len, TF_TensorByteSize(tensor));
	if (data != NULL && len != 0)
		memcpy(tensor_data, data, len);

	return tensor;
}

struct tf_s {

	INTERFACE(nlop_data_t);

	int nr_inputs;
	int nr_outputs;

	TF_Session* sess;
	TF_Status* status;
	TF_Graph* graph;

	TF_Tensor **input_tensors;
	TF_Tensor **output_tensors;
	TF_Tensor **grad_tensors;
	TF_Tensor **grad_ys_tensors;

	struct TF_Output *inputs_op;
	struct TF_Output *outputs_op;
	struct TF_Output *grad_op;
	struct TF_Output *grad_ys_op;

	int *nr_out_dim;
	int *nr_in_dim;
	int *nr_grad_dim;
	int *nr_grad_ys_dim;

	int64_t **out_dims_tf;
	int64_t **in_dims_tf;
	int64_t **grad_dims_tf;
	int64_t **grad_ys_dims_tf;

	bool use_adjoint;

};

DEF_TYPEID(tf_s);

static void float2cplx(unsigned int D, const long *dims, complex float* optr, float* iptr)
{
	long chn_dims[D];
	md_copy_dims(D, chn_dims, dims);
	chn_dims[D-1] = 1;

	long offset = md_calc_size(D, chn_dims);

	float* real = md_alloc(D, chn_dims, FL_SIZE);
	float* imag = md_alloc(D, chn_dims, FL_SIZE);

	md_copy(D, chn_dims, real, iptr, FL_SIZE);
	md_copy(D, chn_dims, imag, iptr+offset, FL_SIZE);

	md_zcmpl(D, chn_dims, optr, real, imag);
	
	free(real);
	free(imag);
}


static void cplx2float(unsigned int D, const long *dims, float* optr, complex float* iptr)
{
	long chn_dims[D];
	md_copy_dims(D, chn_dims, dims);
	chn_dims[D-1] = 1;

	long offset = md_calc_size(D, chn_dims);

	float* real = md_alloc(D, chn_dims, FL_SIZE);
	float* imag = md_alloc(D, chn_dims, FL_SIZE);
	
	md_real(D, chn_dims, real, iptr);
	md_imag(D, chn_dims, imag, iptr);

	md_copy(D, chn_dims, optr + 0, real, FL_SIZE);
	md_copy(D, chn_dims, optr + offset, imag, FL_SIZE);

	free(real);
	free(imag);
}

static void tf_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	auto data = CAST_DOWN(tf_s, _data);
	assert(data->nr_inputs + data->nr_outputs == N);

	for (int i = 0; i < data->nr_inputs; i++)
	{
		//float* tmp = md_alloc(data->nr_in_dim[i], data->in_dims_tf[i], FL_SIZE);
		//cplx2float(data->nr_in_dim[i], data->in_dims_tf[i], tmp, args[i + data->nr_outputs]);
		md_copy(data->nr_in_dim[i], data->in_dims_tf[i], TF_TensorData(data->input_tensors[i]), args[i + data->nr_outputs], FL_SIZE);
	}
		

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ data->inputs_op, data->input_tensors, data->nr_inputs,
				/* Output tensors */ data->outputs_op, data->output_tensors, data->nr_outputs,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);	
	
	for (int i = 0; i < data->nr_outputs; i++)
		md_copy(data->nr_out_dim[i], data->out_dims_tf[i], args[i], TF_TensorData(data->output_tensors[i]), FL_SIZE);
	
		
}

static void tf_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(tf_s, _data);
	error("NOT IMPLEMENTED");
	UNUSED(dst);
	UNUSED(src);
	UNUSED(o);
	UNUSED(i);
}

static void tf_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(tf_s, _data);

	md_copy(data->nr_grad_ys_dim[i], data->grad_ys_dims_tf[i], TF_TensorData(data->grad_ys_tensors[i]), src, FL_SIZE);

	struct TF_Output *inp_ops = (struct TF_Output *)malloc(sizeof(TF_Output) * 2);
	
	inp_ops[0] = data->inputs_op[i];
	inp_ops[1] = data->grad_ys_op[i];

	struct TF_Tensor **inp_tensors = (struct TF_Tensor**)malloc(sizeof(struct TF_Tensor *) * 2);
	inp_tensors[0] = data->input_tensors[i];
	inp_tensors[1] = data->grad_ys_tensors[i];

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ inp_ops, inp_tensors, 2,
				/* Output tensors */ &(data->grad_op[o]), &(data->grad_tensors[o]), 1,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);
	
	
	md_copy(data->nr_grad_dim[o], data->grad_dims_tf[o], dst, TF_TensorData(data->grad_tensors[o]), FL_SIZE);
}


static void tf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tf_s, _data);
	for(int i = 0; i < data->nr_inputs; i++)
		TF_DeleteTensor(data->input_tensors[i]);
}

const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path, bool use_adjoint, bool to_restore)
{
	PTR_ALLOC(struct tf_s, data);
	SET_TYPEID(tf_s, data);

	// load graph, restore session
	char graph_path[strlen(path) + 4];
	sprintf(graph_path, "%s.pb", path);
	const char* cpkt_path = path;

	TF_Buffer* graph_def = read_graph(graph_path);

	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();

	load_graph(graph_def, graph, status);
	char gpu_config_path[strlen(path)+8];
	sprintf(gpu_config_path, "%s_gpu_id", path);
	
	uint8_t* config_char = (uint8_t*) malloc(20*sizeof(uint8_t));
	read_gpu_config(gpu_config_path, config_char);

	TF_Session* sess = create_session(graph, status, config_char, false);
	if(to_restore)
		restore_sess(graph, status, sess, cpkt_path);

	free(config_char);

	data->sess = sess;
	data->status = status;
	data->graph = graph;

	data->nr_inputs = nr_inputs;
	data->nr_outputs = nr_outputs;
	data->use_adjoint = use_adjoint;
	
	/*** handle outputs and grad_ys ***/

	// outputs
	int OO = nr_outputs;
	int ON = 1;
	int ON_arr[OO];

	PTR_ALLOC(struct TF_Output[data->nr_outputs], outputs_op);
	data->outputs_op = PTR_PASS(outputs_op);

	PTR_ALLOC(struct TF_Tensor*[data->nr_outputs], output_tensors);
	data->output_tensors = PTR_PASS(output_tensors);
	
	PTR_ALLOC(int[data->nr_outputs], nr_out_dim);
	data->nr_out_dim = PTR_PASS(nr_out_dim);

	PTR_ALLOC(int64_t*[data->nr_outputs], out_dims_tf);
	data->out_dims_tf = PTR_PASS(out_dims_tf);
	
	// grad_ys_
	if(data->use_adjoint){
		
		PTR_ALLOC(struct TF_Output[data->nr_outputs], grad_ys_op);
		data->grad_ys_op = PTR_PASS(grad_ys_op);

		PTR_ALLOC(struct TF_Tensor*[data->nr_outputs], grad_ys_tensors);
		data->grad_ys_tensors = PTR_PASS(grad_ys_tensors);

		PTR_ALLOC(int[data->nr_outputs], nr_grad_ys_dim);
		data->nr_grad_ys_dim = PTR_PASS(nr_grad_ys_dim);

		PTR_ALLOC(int64_t*[data->nr_outputs], grad_ys_dims_tf);
		data->grad_ys_dims_tf = PTR_PASS(grad_ys_dims_tf);
	}

	for (int i = 0; i < data->nr_outputs; i++){

		char out_name[20];
		sprintf(out_name, "output_%d", i);

		data->outputs_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0};
		data->nr_out_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->outputs_op[i], data->status);

		ON_arr[i] = data->nr_out_dim[i];
		ON = MAX(ON, ON_arr[i]);

		PTR_ALLOC(int64_t[data->nr_out_dim[i]], nr_out_dim);
		data->out_dims_tf[i] = PTR_PASS(nr_out_dim);
		
		TF_GraphGetTensorShape(data->graph, data->outputs_op[i], data->out_dims_tf[i], data->nr_out_dim[i], data->status);

		data->output_tensors[i] = TF_AllocateTensor(TF_FLOAT, 
													data->out_dims_tf[i],
													data->nr_out_dim[i],
													product(data->nr_out_dim[i], data->out_dims_tf[i]) * FL_SIZE);

		if(data->use_adjoint){
			char grad_ys_name[20];
			sprintf(grad_ys_name, "grad_ys_%d", i); 
			
			data->grad_ys_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, grad_ys_name), 0};
			data->nr_grad_ys_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->grad_ys_op[i], data->status);

			PTR_ALLOC(int64_t[data->nr_grad_ys_dim[i]], nr_grad_ys_dim);
			data->grad_ys_dims_tf[i] = PTR_PASS(nr_grad_ys_dim);

			TF_GraphGetTensorShape(data->graph, data->grad_ys_op[i], data->grad_ys_dims_tf[i], data->nr_grad_ys_dim[i], data->status);
		
			data->grad_ys_tensors[i] = TF_AllocateTensor(TF_FLOAT,
													data->grad_ys_dims_tf[i],
													data->nr_grad_ys_dim[i],
													product(data->nr_grad_ys_dim[i], data->grad_ys_dims_tf[i]) * FL_SIZE);
		}
		
												
	}

	// handle inputs and grad
	int II = nr_inputs;
	int IN = 1;
	int IN_arr[II];

	PTR_ALLOC(struct TF_Output[data->nr_inputs], inputs_op);
	data->inputs_op = PTR_PASS(inputs_op);

	PTR_ALLOC(struct TF_Tensor*[data->nr_inputs], input_tensors);
	data->input_tensors = PTR_PASS(input_tensors);
	
	PTR_ALLOC(int[data->nr_inputs], nr_in_dim);
	data->nr_in_dim = PTR_PASS(nr_in_dim);

	PTR_ALLOC(int64_t *[data->nr_inputs], in_dims_tf);
	data->in_dims_tf = PTR_PASS(in_dims_tf);

	if(data->use_adjoint){

		PTR_ALLOC(struct TF_Output[data->nr_inputs], grad_op);
		data->grad_op = PTR_PASS(grad_op);

		PTR_ALLOC(struct TF_Tensor*[data->nr_inputs], grad_tensors);
		data->grad_tensors = PTR_PASS(grad_tensors);

		PTR_ALLOC(int[data->nr_inputs], nr_grad_dim);
		data->nr_grad_dim = PTR_PASS(nr_grad_dim);

		PTR_ALLOC(int64_t *[data->nr_inputs], grad_dims_tf);
		data->grad_dims_tf = PTR_PASS(grad_dims_tf);
	}
	

	for (int i = 0; i < data->nr_inputs; i++){

		// inputs
		char in_name[20];
		sprintf(in_name, "input_%d", i);

		data->inputs_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0};
		data->nr_in_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->inputs_op[i], data->status);

		IN_arr[i] = data->nr_in_dim[i];
		IN = MAX(IN, IN_arr[i]);

		PTR_ALLOC(int64_t[data->nr_in_dim[i]], in_dims_tf);
		data->in_dims_tf[i] = PTR_PASS(in_dims_tf);
		
		TF_GraphGetTensorShape(data->graph, data->inputs_op[i], data->in_dims_tf[i], data->nr_in_dim[i], data->status);
		debug_print_dims(DP_DEBUG3, data->nr_in_dim[i], data->in_dims_tf[i]);
		data->input_tensors[i] = TF_AllocateTensor(TF_FLOAT,
													data->in_dims_tf[i],
													data->nr_in_dim[i],
													product(data->nr_in_dim[i], data->in_dims_tf[i]) * FL_SIZE);

		//grads
		if(data->use_adjoint){
			char grad_name[20];
			sprintf(grad_name, "grad_%d", i);
			
			data->grad_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, grad_name), 0};
			data->nr_grad_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->grad_op[i], data->status);

			PTR_ALLOC(int64_t[data->nr_grad_dim[i]], grad_dims_tf);
			data->grad_dims_tf[i] = PTR_PASS(grad_dims_tf);

			TF_GraphGetTensorShape(data->graph, data->grad_op[i], data->grad_dims_tf[i], data->nr_grad_dim[i], data->status);

			data->grad_tensors[i] = TF_AllocateTensor(TF_FLOAT,
													data->grad_dims_tf[i],
													data->nr_grad_dim[i],
													product(data->nr_grad_dim[i], data->grad_dims_tf[i]) * FL_SIZE);
		}

	}

	long nl_odims[OO][ON];
	long nl_idims[II][IN];

	for (int i = 0; i< nr_outputs; i++) {

		char out_name[20]; sprintf(out_name, "output_%d", i);

		int64_t dims_tf[ON_arr[i]];
		TF_GraphGetTensorShape(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0}, dims_tf, ON_arr[i], data->status);

		for (int j = 0; j < ON; j++)
			nl_odims[i][j] = (j < ON_arr[i] - 1) ? dims_tf[j] : 1;
	}

	for (int i = 0; i< nr_inputs; i++) {

		char in_name[20]; sprintf(in_name, "input_%d", i);

		int64_t dims_tf[IN_arr[i]];
		TF_GraphGetTensorShape(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0}, dims_tf, IN_arr[i], data->status);

		for (int j = 0; j < IN; j++)
			nl_idims[i][j] = (j < IN_arr[i]-1) ? dims_tf[j] : 1;
	}

	nlop_der_fun_t deriv[II][OO];
	nlop_der_fun_t adjoint[II][OO];
	nlop_der_fun_t normal[II][OO];
	nlop_p_fun_t norm_inv[II][OO];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {
			if(data->use_adjoint){
				deriv[i][o] = tf_der;
				adjoint[i][o] = tf_adj;
			}
			else{
				deriv[i][o] = NULL;
				adjoint[i][o] = NULL;
			}
			normal[i][o] = NULL;
			norm_inv[i][o] = NULL;
		}

	const struct nlop_s* result = nlop_generic_create(	OO, ON, nl_odims, II, IN, nl_idims,
								CAST_UP(PTR_PASS(data)), tf_forward, deriv, adjoint, normal, norm_inv, tf_del);
/*
	for (int i = 0; i < II; i++) {

			auto iov = nlop_generic_domain(result, i);
			result = nlop_reshape_in_F(result, i, MAX(IN_arr[i] - 1, 1), iov->dims);
		}

	for (int o = 0; o < OO; o++) {

			auto iov = nlop_generic_codomain(result, o);
			result = nlop_reshape_out_F(result, o, MAX(ON_arr[o] - 1, 1), iov->dims);
		}
*/
	return result;
}


extern const struct TF_Tensor** get_input_tensor(struct nlop_s* op)
{
	
	nlop_data_t* data = nlop_get_data(op);
	
	struct tf_s* tf_data = CAST_DOWN(tf_s, data);
	
	return tf_data->input_tensors;
}
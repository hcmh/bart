
#include <complex.h>
#include <stdio.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

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
		error("ERROR: Loading graph failed: %s\n", TF_Message(status));

	debug_printf(DP_INFO, "Graph loaded!\n");
}

// function to create session
static TF_Session* create_session(TF_Graph* graph, TF_Status* status, bool test)
{
	TF_SessionOptions* opt = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, opt, status);
	TF_DeleteSessionOptions(opt);
	if (TF_GetCode(status) != TF_OK)
		error("ERROR: Unable to create session %s\n", TF_Message(status));

	debug_printf(DP_INFO, "Session created.\n");

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
			error("ERROR: Unable to run init_op: %s\n", TF_Message(status));


		debug_printf(DP_INFO, "Session initialized.\n");
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
		error("ERROR: something wrong with encoding: %s", TF_Message(status));

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



	debug_printf(DP_INFO, "Session restored.\n");

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
};

DEF_TYPEID(tf_s);


static void tf_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	auto data = CAST_DOWN(tf_s, _data);
	assert(data->nr_inputs + data->nr_outputs == N);

	struct TF_Output run_inputs[data->nr_inputs];
	TF_Tensor* input_tensors[data->nr_inputs];

	for (int i = 0; i < data->nr_inputs; i++){

		char in_name[20];
		sprintf(in_name, "input_%d", i);

		run_inputs[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0};
		int nr_dim = TF_GraphGetTensorNumDims(data->graph, run_inputs[i], data->status);

		int64_t dims_tf[nr_dim];
		TF_GraphGetTensorShape(data->graph, run_inputs[i], dims_tf, nr_dim, data->status);
		
		input_tensors[i] = TF_AllocateTensor(TF_FLOAT, dims_tf, nr_dim,  product(nr_dim, dims_tf) * FL_SIZE);
		
		md_copy(nr_dim, dims_tf, TF_TensorData(input_tensors[i]), args[i + data->nr_outputs], FL_SIZE);

	}

	struct TF_Output run_outputs[data->nr_outputs];
	TF_Tensor* output_tensors[data->nr_outputs];

	for (int i = 0; i < data->nr_outputs; i++){

		char out_name[20];
		sprintf(out_name, "output_%d", i);

		run_outputs[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0};
		int nr_dim = TF_GraphGetTensorNumDims(data->graph, run_outputs[i], data->status);

		int64_t dims_tf[nr_dim];
		TF_GraphGetTensorShape(data->graph, run_outputs[i], dims_tf, nr_dim, data->status);

		output_tensors[i] = TF_AllocateTensor(TF_FLOAT, dims_tf, nr_dim, product(nr_dim, dims_tf) * FL_SIZE);
		
	}


	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ run_inputs, input_tensors, data->nr_inputs,
				/* Output tensors */ run_outputs, output_tensors, data->nr_outputs,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);	

	
	// Delete tf tensor

	for (int i = 0; i < data->nr_inputs; i++)
		TF_DeleteTensor(input_tensors[i]);

	for (int i = 0; i < data->nr_outputs; i++){

		char out_name[20];
		sprintf(out_name, "output_%d", i);

		run_outputs[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0};
		int nr_dim = TF_GraphGetTensorNumDims(data->graph, run_outputs[i], data->status);

		int64_t dims_tf[nr_dim];
		TF_GraphGetTensorShape(data->graph, run_outputs[i], dims_tf, nr_dim, data->status);
		md_copy(nr_dim, dims_tf, args[i], TF_TensorData(output_tensors[i]), FL_SIZE);
		TF_DeleteTensor(output_tensors[i]);
	}
		

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

	// grad_ys_
	char in_name[20];
	sprintf(in_name, "grad_ys_%d", o); // o corresponds to output of forward model

	struct TF_Output run_input = {TF_GraphOperationByName(data->graph, in_name), 0};
	int nr_in_dim = TF_GraphGetTensorNumDims(data->graph, run_input, data->status);

	int64_t in_dims_tf[nr_in_dim];
	TF_GraphGetTensorShape(data->graph, run_input, in_dims_tf, nr_in_dim, data->status);

	TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, in_dims_tf, nr_in_dim, product(nr_in_dim, in_dims_tf) * FL_SIZE);
	
	md_copy(nr_in_dim, in_dims_tf, TF_TensorData(input_tensor), src, FL_SIZE);

	// grad_
	char out_name[20];
	sprintf(out_name, "grad_%d", i);

	struct TF_Output run_output = {TF_GraphOperationByName(data->graph, out_name), 0};
	int nr_out_dim = TF_GraphGetTensorNumDims(data->graph, run_output, data->status);

	int64_t out_dims_tf[nr_out_dim];
	TF_GraphGetTensorShape(data->graph, run_output, out_dims_tf, nr_out_dim, data->status);

	TF_Tensor* output_tensor = TF_AllocateTensor(TF_FLOAT, out_dims_tf, nr_out_dim, product(nr_out_dim, out_dims_tf) * FL_SIZE);	

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ &run_input, &input_tensor, 1,
				/* Output tensors */ &run_output, &output_tensor, 1,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);
	
	// Delete tf tensor

	TF_DeleteTensor(input_tensor);


	md_copy(nr_out_dim, out_dims_tf, dst, TF_TensorData(output_tensor), FL_SIZE);
	TF_DeleteTensor(output_tensor);

	

}


static void tf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tf_s, _data);
}


const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path)
{
	PTR_ALLOC(struct tf_s, data);
	SET_TYPEID(tf_s, data);

	char graph_path[strlen(path) + 4];
	sprintf(graph_path, "%s.pb", path);
	const char* cpkt_path = path;

	TF_Buffer* graph_def = read_graph(graph_path);

	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();

	load_graph(graph_def, graph, status);
	TF_Session* sess = create_session(graph, status, false);
	restore_sess(graph, status, sess, cpkt_path);

	data->sess = sess;
	data->status = status;
	data->graph = graph;

	data->nr_inputs = nr_inputs;
	data->nr_outputs = nr_outputs;


	int OO = nr_outputs;
	int ON = 1;
	int ON_arr[OO];

	for (int i = 0; i< nr_outputs; i++) {

		char out_name[20]; sprintf(out_name, "output_%d", i);
		ON_arr[i] = TF_GraphGetTensorNumDims(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0}, data->status);
		ON = MAX(ON, ON_arr[i]);
	}


	int II = nr_inputs;
	int IN = 1;
	int IN_arr[II];

	for (int i = 0; i< nr_inputs; i++) {

		char in_name[20]; sprintf(in_name, "input_%d", i);
		IN_arr[i] = TF_GraphGetTensorNumDims(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0}, data->status);
		IN = MAX(IN, IN_arr[i]);
	}

	long nl_odims[OO][ON];
	long nl_idims[II][IN];

	for (int i = 0; i< nr_outputs; i++) {

		char out_name[20]; sprintf(out_name, "output_%d", i);

		int64_t dims_tf[ON_arr[i]];
		TF_GraphGetTensorShape(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0}, dims_tf, ON_arr[i], data->status);

		for (int j = 0; j < ON; j++)
			nl_odims[i][j] = (j < ON_arr[i] - 1) ? dims_tf[ON_arr[i] - 2 - j] : 1;
	}

	for (int i = 0; i< nr_outputs; i++) {

		char in_name[20]; sprintf(in_name, "input_%d", i);

		int64_t dims_tf[IN_arr[i]];
		TF_GraphGetTensorShape(data->graph, (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0}, dims_tf, IN_arr[i], data->status);

		for (int j = 0; j < IN; j++)
			nl_idims[i][j] = (j < IN_arr[i]-1) ? dims_tf[IN_arr[i] -2 - j] : 1;
	}

	nlop_der_fun_t deriv[II][OO];
	nlop_der_fun_t adjoint[II][OO];
	nlop_der_fun_t normal[II][OO];
	nlop_p_fun_t norm_inv[II][OO];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			deriv[i][o] = tf_der;
			adjoint[i][o] = tf_adj;
			normal[i][o] = NULL;
			norm_inv[i][o] = NULL;
		}

	const struct nlop_s* result = nlop_generic_create(	OO, ON, nl_odims, II, IN, nl_idims,
								CAST_UP(PTR_PASS(data)), tf_forward, deriv, adjoint, normal, norm_inv, tf_del);

	for (int i = 0; i < II; i++) {

			auto iov = nlop_generic_domain(result, i);
			result = nlop_reshape_in_F(result, i, MAX(IN_arr[i] - 1, 1), iov->dims);
		}

	for (int o = 0; o < OO; o++) {

			auto iov = nlop_generic_codomain(result, o);
			result = nlop_reshape_out_F(result, o, MAX(ON_arr[o] - 1, 1), iov->dims);
		}

	return result;
}

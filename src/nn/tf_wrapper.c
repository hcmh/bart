
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

	TF_Tensor **input_tensors;
	TF_Tensor **output_tensors;
	TF_Tensor *grad_tensors;
	TF_Tensor *grad_ys_tensors;

	struct TF_Output *inputs_op;
	struct TF_Output *outputs_op;
	struct TF_Output grad_op;
	struct TF_Output grad_ys_op;

	int *nr_out_dim;
	int *nr_in_dim;
	int nr_grad_dim;
	int nr_grad_ys_dim;

	int64_t **out_dims_tf;
	int64_t **in_dims_tf;
	int64_t *grad_dims_tf;
	int64_t *grad_ys_dims_tf;

};

DEF_TYPEID(tf_s);


static void tf_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	auto data = CAST_DOWN(tf_s, _data);
	assert(data->nr_inputs + data->nr_outputs == N);

	for (int i = 0; i < data->nr_inputs; i++)
		md_copy(data->nr_in_dim[i], data->in_dims_tf[i], TF_TensorData(data->input_tensors[i]), args[i + data->nr_outputs], FL_SIZE);

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ data->inputs_op, data->input_tensors, data->nr_inputs,
				/* Output tensors */ data->outputs_op, data->output_tensors, data->nr_outputs,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);	
	
	// Delete tf tensor

	//for (int i = 0; i < data->nr_inputs; i++)
	//	TF_DeleteTensor(data->input_tensors[i]);

	for (int i = 0; i < data->nr_outputs; i++){
		md_copy(data->nr_out_dim[i], data->out_dims_tf[i], args[i], TF_TensorData(data->output_tensors[i]), FL_SIZE);
		TF_DeleteTensor(data->output_tensors[i]);
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

	data->grad_ys_op = (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0};
	data->nr_grad_ys_dim = TF_GraphGetTensorNumDims(data->graph, data->grad_ys_op, data->status);

	data->grad_ys_dims_tf = (int64_t*)malloc(sizeof(int64_t) * data->nr_grad_ys_dim);
	TF_GraphGetTensorShape(data->graph, data->grad_ys_op, data->grad_ys_dims_tf, data->nr_grad_ys_dim, data->status);

	data->grad_ys_tensors = TF_AllocateTensor(TF_FLOAT,
												data->grad_ys_dims_tf,
												data->nr_grad_ys_dim,
												product(data->nr_grad_ys_dim, data->grad_ys_dims_tf) * FL_SIZE);
	
	md_copy(data->nr_grad_ys_dim, data->grad_ys_dims_tf, TF_TensorData(data->grad_ys_tensors), src, FL_SIZE);

	// grad_
	char out_name[20];
	sprintf(out_name, "grad_%d", i);

	data->grad_op = (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0};
	data->nr_grad_dim = TF_GraphGetTensorNumDims(data->graph, data->grad_op, data->status);	

	data->grad_dims_tf = (int64_t*)malloc(sizeof(int64_t) * data->nr_grad_dim);

	TF_GraphGetTensorShape(data->graph, data->grad_op, data->grad_dims_tf, data->nr_grad_dim, data->status);

	data->grad_tensors = TF_AllocateTensor(TF_FLOAT, 
											data->grad_dims_tf,
											data->nr_grad_dim,
											product(data->nr_grad_dim, data->grad_dims_tf) * FL_SIZE);	
	struct TF_Output *inp_ops = (struct TF_Output *)malloc(sizeof(TF_Output) * 2);
	
	inp_ops[0] = data->inputs_op[0];
	inp_ops[1] = data->grad_ys_op;

	struct TF_Tensor **inp_tensors = (struct TF_Tensor**)malloc(sizeof(struct TF_Tensor *) * 2);
	inp_tensors[0] = data->input_tensors[0];
	inp_tensors[1] = data->grad_ys_tensors;

	TF_SessionRun(data->sess,
				/* RunOptions */ NULL,
				/* Input tensors */ inp_ops, inp_tensors, 2,
				/* Output tensors */ &(data->grad_op), &(data->grad_tensors), 1,
				/* Target operations */ NULL, 0,
				/* RunMetadata */ NULL,
				/* Output status */ data->status);
	
	// Delete tf tensor

	TF_DeleteTensor(inp_tensors[0]);
	TF_DeleteTensor(inp_tensors[1]);


	md_copy(data->nr_grad_dim, data->grad_dims_tf, dst, TF_TensorData(data->grad_tensors), FL_SIZE);
	TF_DeleteTensor(data->grad_tensors);
}


static void tf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tf_s, _data);
}


const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path)
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
	TF_Session* sess = create_session(graph, status, false);
	restore_sess(graph, status, sess, cpkt_path);

	data->sess = sess;
	data->status = status;
	data->graph = graph;

	data->nr_inputs = nr_inputs;
	data->nr_outputs = nr_outputs;


	// handle outputs
	int OO = nr_outputs;
	int ON = 1;
	int ON_arr[OO];

	data->outputs_op = (struct TF_Output*)malloc(sizeof(struct TF_Output) * data->nr_outputs);
	data->output_tensors = (TF_Tensor**)malloc(sizeof(TF_Tensor *) * data->nr_outputs);
	
	data->nr_out_dim = (int *)malloc(sizeof(int) * data->nr_outputs);
	data->out_dims_tf = (int64_t**)malloc(sizeof(int64_t *) * data->nr_outputs);

	for (int i = 0; i < data->nr_outputs; i++){

		char out_name[20];
		sprintf(out_name, "output_%d", i);

		data->outputs_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, out_name), 0};
		data->nr_out_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->outputs_op[i], data->status);

		ON_arr[i] = data->nr_out_dim[i];
		ON = MAX(ON, ON_arr[i]);

		data->out_dims_tf[i] = (int64_t *)malloc(sizeof(int64_t) * data->nr_out_dim[i]);
		
		TF_GraphGetTensorShape(data->graph, data->outputs_op[i], data->out_dims_tf[i], data->nr_out_dim[i], data->status);

		data->output_tensors[i] = TF_AllocateTensor(TF_FLOAT, 
													data->out_dims_tf[i],
													data->nr_out_dim[i],
													product(data->nr_out_dim[i], data->out_dims_tf[i]) * FL_SIZE);
		
	}

	// handle inputs
	int II = nr_inputs;
	int IN = 1;
	int IN_arr[II];

	data->inputs_op = (struct TF_Output*)malloc(sizeof(struct TF_Output) * data->nr_inputs);
	data->input_tensors = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * data->nr_inputs);
	
	data->nr_in_dim = (int *)malloc(sizeof(int) * data->nr_inputs);
	data->in_dims_tf = (int64_t**)malloc(sizeof(int64_t *) * data->nr_inputs);

	for (int i = 0; i < data->nr_inputs; i++){

		char in_name[20];
		sprintf(in_name, "input_%d", i);

		data->inputs_op[i] = (struct TF_Output){TF_GraphOperationByName(data->graph, in_name), 0};
		data->nr_in_dim[i] = TF_GraphGetTensorNumDims(data->graph, data->inputs_op[i], data->status);

		IN_arr[i] = data->nr_in_dim[i];
		IN = MAX(IN, IN_arr[i]);

		data->in_dims_tf[i] = (int64_t*)malloc(sizeof(int64_t) * data->nr_in_dim[i]);
		
		TF_GraphGetTensorShape(data->graph, data->inputs_op[i], data->in_dims_tf[i], data->nr_in_dim[i], data->status);
		
		data->input_tensors[i] = TF_AllocateTensor(TF_FLOAT,
													data->in_dims_tf[i],
													data->nr_in_dim[i],
													product(data->nr_in_dim[i], data->in_dims_tf[i]) * FL_SIZE);

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

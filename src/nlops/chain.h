struct nlop_s;

extern struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_combine_FF(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii);
extern struct nlop_s* nlop_link_F(const struct nlop_s* x, int oo, int ii);
extern struct nlop_s* nlop_permute_inputs(const struct nlop_s* x, int I2, const int perm[__VLA(I2)]);
extern struct nlop_s* nlop_permute_outputs(const struct nlop_s* x, int O2, const int perm[__VLA(O2)]);
extern struct nlop_s* nlop_permute_inputs_F(const struct nlop_s* x, int I2, const int perm[__VLA(I2)]);
extern struct nlop_s* nlop_permute_outputs_F(const struct nlop_s* x, int O2, const int perm[__VLA(O2)]);
extern struct nlop_s* nlop_dup(const struct nlop_s* x, int a, int b);
extern struct nlop_s* nlop_dup_F(const struct nlop_s* x, int a, int b);
extern struct nlop_s* nlop_destack(const struct nlop_s* x, int a, int b, unsigned long stack_dim);//deprecated, equivalent to nlop_stack_inputs
extern struct nlop_s* nlop_destack_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim);
extern struct nlop_s* nlop_stack_inputs(const struct nlop_s* x, int a, int b, unsigned long stack_dim);
extern struct nlop_s* nlop_stack_inputs_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim);
extern struct nlop_s* nlop_stack_outputs(const struct nlop_s* x, int a, int b, unsigned long stack_dim);
extern struct nlop_s* nlop_stack_outputs_F(const struct nlop_s* x, int a, int b, unsigned long stack_dim);
extern struct nlop_s* nlop_shift_input(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_input_F(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_output(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_output_F(const struct nlop_s* x, int new_index, int old_index);
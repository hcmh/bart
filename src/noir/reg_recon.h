
struct noir_conf_s;
struct iter3_irgnm_conf;
struct nlop_s;
struct linop_s;
struct opt_reg_s;
struct operator_p_s;
struct lsqr_conf;

#ifndef DIMS
#define DIMS 16
#endif


struct irgnm_reg_conf{

    struct iter3_irgnm_conf* irgnm_conf;

    struct opt_reg_s* ropts;
    unsigned int algo;

    float step;
    float rho;
    unsigned int maxiter;
    float tol;
    unsigned int max_outiter;
    unsigned int shift_mode;
    struct iter_admm_conf *admm_conf_reg;
    struct lsqr_conf *lsqr_conf_reg;
    
};

extern const struct operator_p_s* reg_pinv_op_create(struct irgnm_reg_conf* conf, const long dims[DIMS], struct nlop_s* nlop, const struct operator_p_s** thresh_ops, const struct linop_s** trafos);
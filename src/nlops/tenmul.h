
struct nlop_s;
struct nlop_s* nlop_tenmul_create2(int N, const long dims[N], const long dstr[N],
		const long istr1[N], const long istr[N], bool use_gpu);


struct nlop_s* nlop_tenmul_create(int N, const long odim[N], const long idim1[N], const long idims2[N], bool use_gpu);

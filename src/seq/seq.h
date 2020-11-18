


enum seq_event_type { SEQ_EVENT_PULSE, SEQ_EVENT_GRADIENT, SEQ_EVENT_ADC, SEQ_EVENT_WAIT };


struct seq_pulse {

	double fa;
	double phase;
};

struct seq_gradient {

	double ampl[3];
};

struct seq_adc {

	double freq;
	double phase;
};

struct seq_wait {

};

struct seq_event {

	double start;
	double mid;
	double end;

	enum seq_event_type type;

	const struct seq_event* dependency;

	union {

		struct seq_pulse pulse;
		struct seq_gradient grad;
		struct seq_adc adc;
		struct seq_wait wait;
	};
};


extern void seq_compute_gradients(int M, float gradients[M][3], double dt, int N, const struct seq_event ev[N]);
extern void seq_sample(double m[3], double t, int N, const struct seq_event ev[N]);


struct seq_flash_conf {

	double TR;	// [s]
	double TE;	// [s]
	double FA;	// [deg]
	double BW_adc;	// [Hz/pixel]
	double BW_rf;	// [Hz]

	double res[3];	// pixel size [m]
	double FOV[3];	// FOV [m]
	double pos[3];	// slice center position relative to isocenter [m]
};

struct seq_system {

	const char* name;
	double inv_slew_rate;
	double max_grad_ampl;
	double min_coil_lead;
	double min_dur_readout_rf;
	double sample_time;
	double raster_time;
};

extern struct seq_system seq_sys_skyra;
extern struct seq_system seq_sys_skyra_whisper;


extern struct seq_flash_conf seq_flash_defaults;
extern void seq_flash(int N, struct seq_event[N], struct seq_flash_conf conf, const struct seq_system* sys);


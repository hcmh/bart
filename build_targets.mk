# Main build targets
#
TBASE=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati vec poly index
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexp
TNUM=fft fftmod fftshift noise bench threshold conv rss filter mandelbrot wavelet window var std fftrot
TRECO=pics pocsense sqpics itsense nlinv T1fun mdbT1 mdbT2 modbloch nufft rof tgv sake wave lrmatrix estdims estshift estdelay wavepsf wshfl hornschunck ncsense kmat power approx kernel dcnn rtreco rtnlinv
TCALIB=ecalib ecaltwo caldir walsh cc ccapply calmat svd estvar whiten ssa bin cordelay laplace kmeans
TMRI=homodyne poisson twixread fakeksp umgread looklocker
TSIM=phantom traj upat bloch
TIO=toimg dcmread

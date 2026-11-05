#!/bin/bash
# Run standalone tools on one-pair (and control two-pair) real subsets for a given code tree.
SP=$1; S=$2; X=$3; code=$4; pp=$5; src=$6; label=$7
wd=$X/${label}_$code; mkdir -p $wd; cd $wd
cp -r $S/runs/$src/inputs . ; cp $S/runs/$src/timeseries.h5 $S/runs/$src/velocity.h5 $S/runs/$src/temporalCoherence.h5 $S/runs/$src/maskTempCoh.h5 . 2>/dev/null
export MPLBACKEND=Agg HOME=$SP/fakehome OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PATH=$SP/venv/bin:$PATH
unset PYTHONPATH; [ -n "$pp" ] && export PYTHONPATH=$pp
run() { n=$1; shift; "$@" > $n.log 2>&1; rc=$?; last=$(grep -E '^[A-Za-z.]*(Error|Exception)' $n.log | tail -1 | cut -c1-90); where=$(grep -E '^  File ".*mintpy' $n.log | tail -1 | sed -E 's#.*/mintpy/([^"]+)", line ([0-9]+).*#\1:\2#'); echo "$label $code $n rc=$rc $where $last"; }
run tsview_fit       tsview.py timeseries.h5 --yx 10 10 --nodisplay --noverbose
run tsview_confint   tsview.py timeseries.h5 --yx 10 10 --plot-model-conf-int --nodisplay --noverbose
run tsview_nomodel   tsview.py timeseries.h5 --yx 10 10 --nomodel --nodisplay --noverbose
run plot_cohmat      plot_coherence_matrix.py inputs/ifgramStack.h5 --yx 10 10 --noverbose --nodisplay
run temporal_avg     temporal_average.py inputs/ifgramStack.h5 -d coherence -o tavg.h5
run timeseries_rms   timeseries_rms.py timeseries.h5
run reference_date   reference_date.py timeseries.h5 --ref-date minRMS -o ts_refd.h5
run ifg_recon        ifgram_reconstruction.py timeseries.h5 -r inputs/ifgramStack.h5 -o recon.h5
run dem_error_cli    dem_error.py timeseries.h5 -g inputs/geometryRadar.h5 -o ts_dem.h5
run ts2vel_cli       timeseries2velocity.py timeseries.h5 -o vel_cli.h5
run ts2vel_step      timeseries2velocity.py timeseries.h5 --poly 1 -o vel_noref.h5 --ref-date first
run view_ts          view.py timeseries.h5 --nodisplay --noverbose
run info_ts          info.py timeseries.h5 --date

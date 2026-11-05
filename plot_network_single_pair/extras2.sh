#!/bin/bash
# Scan every general-purpose CLI tool that accepts a stack or a time series.
SP=$1; S=$2; X=$3; code=$4; pp=$5; src=$6; label=$7
wd=$X/${label}_$code; mkdir -p $wd; cd $wd
cp -r $S/runs/$src/inputs . ; cp $S/runs/$src/timeseries.h5 $S/runs/$src/velocity.h5 $S/runs/$src/temporalCoherence.h5 $S/runs/$src/maskTempCoh.h5 .
export MPLBACKEND=Agg HOME=$SP/fakehome OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PATH=$SP/venv/bin:$PATH
unset PYTHONPATH; [ -n "$pp" ] && export PYTHONPATH=$pp
run() { n=$1; shift; timeout 120 "$@" > $n.log 2>&1; rc=$?; last=$(grep -E '^[A-Za-z_.]*(Error|Exception|Exit)' $n.log | tail -1 | cut -c1-80); where=$(grep -E '^  File ".*mintpy' $n.log | tail -1 | sed -E 's#.*/mintpy/([^"]+)", line ([0-9]+).*#\1:\2#'); echo "$label $code $n rc=$rc $where $last"; }
I=inputs/ifgramStack.h5; G=inputs/geometryRadar.h5
run multilook_stack   multilook.py $I -r 2 -a 2 -o ml_stack.h5
run multilook_ts      multilook.py timeseries.h5 -r 2 -a 2 -o ml_ts.h5
run subset_stack      subset.py $I -x 0 100 -y 0 100 -o sub_stack.h5
run subset_ts         subset.py timeseries.h5 -x 0 100 -y 0 100 -o sub_ts.h5
run mask_ts           mask.py timeseries.h5 -m maskTempCoh.h5 -o ts_msk.h5
run remove_ramp_ts    remove_ramp.py timeseries.h5 -s linear -o ts_ramp.h5
run spatial_filter_ts spatial_filter.py timeseries.h5 -f lowpass_gaussian -p 3.0 -o ts_sf.h5
run temporal_filter   temporal_filter.py timeseries.h5 -f gaussian -p 0.1 -o ts_tf.h5
run temporal_deriv    temporal_derivative.py timeseries.h5 -o ts_td.h5
run temporal_avg_unw  temporal_average.py $I -d unwrapPhase -o tavg_unw.h5
run genmask_tcoh      generate_mask.py temporalCoherence.h5 -m 0.7 -o m1.h5
run genmask_nonzero   generate_mask.py $I --nonzero -o m2.h5
run add_ts            add.py timeseries.h5 timeseries.h5 -o add.h5
run diff_ts           diff.py timeseries.h5 timeseries.h5 -o diff.h5
run image_math_ts     image_math.py timeseries.h5 '*' 2 -o math.h5
run spatial_avg_ts    spatial_average.py timeseries.h5 --nodisplay
run reference_point   reference_point.py $I -y 10 -x 10
run view_stack        view.py $I unwrapPhase --nodisplay --noverbose
run view_stack_coh    view.py $I coherence --nodisplay --noverbose
run info_stack        info.py $I --date
run plot_transection  plot_transection.py velocity.h5 --start-yx 5 5 --end-yx 20 20 --nodisplay --noverbose
run save_gdal_ts      save_gdal.py timeseries.h5 -d $(info.py timeseries.h5 --date | tail -1) -o ts.tif
run save_roipac_ts    save_roipac.py timeseries.h5 $(info.py timeseries.h5 --date | tail -1) -o ts.unw
run save_roipac_stack save_roipac.py $I $(info.py $I --date | tail -1) -o st.unw
run save_kite_ts      save_kite.py timeseries.h5 -d $(info.py timeseries.h5 --date | tail -1) -g $G -o kite_ts
run save_gbis_vel     save_gbis.py velocity.h5 -g $G -o vel.mat
run save_explorer_ts  save_explorer.py timeseries.h5 -v velocity.h5 -o exp
run closure_bias      closure_phase_bias.py -i $I --nl 3 -a mask -o cpb.h5

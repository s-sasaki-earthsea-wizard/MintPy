#!/bin/bash
# Standalone tools with corrected arguments, on radar and geocoded subsets.
SP=$1; S=$2; X=$3; code=$4; pp=$5; src=$6; label=$7
wd=$X/${label}_$code; mkdir -p $wd; cd $wd
cp -r $S/runs/$src/inputs . ; cp $S/runs/$src/timeseries.h5 $S/runs/$src/velocity.h5 $S/runs/$src/temporalCoherence.h5 $S/runs/$src/maskTempCoh.h5 .
export MPLBACKEND=Agg HOME=$SP/fakehome OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PATH=$SP/venv/bin:$PATH
unset PYTHONPATH; [ -n "$pp" ] && export PYTHONPATH=$pp
run() { n=$1; shift; timeout 120 "$@" > $n.log 2>&1; rc=$?; last=$(grep -E '^[A-Za-z_.]*(Error|Exception|Exit)|^[a-z_0-9.]*: error' $n.log | tail -1 | cut -c1-90); where=$(grep -E '^  File ".*mintpy' $n.log | tail -1 | sed -E 's#.*/mintpy/([^"]+)", line ([0-9]+).*#\1:\2#'); echo "$label $code $n rc=$rc $where $last"; }
I=inputs/ifgramStack.h5; G=$(ls inputs/geometry*.h5 | head -1)
D12=$(info.py $I --date | tail -1); D=$(info.py timeseries.h5 --date | tail -1); D0=$(info.py timeseries.h5 --date | head -1)
run plot_cohmat_time  plot_coherence_matrix.py $I --yx 10 10 --noverbose --nodisplay
run plot_cohmat_index plot_coherence_matrix.py $I --yx 10 10 --ax-fmt index --noverbose --nodisplay
run tsview            tsview.py timeseries.h5 --yx 10 10 --plot-model-conf-int --nodisplay --noverbose
run temporal_filter   temporal_filter.py timeseries.h5 -t 0.1 -o ts_tf.h5
run save_roipac_stack save_roipac.py $I unwrapPhase-$D12 -o st.unw
run save_roipac_ts    save_roipac.py timeseries.h5 ${D0}_$D -o ts.unw
run save_kite_ts      save_kite.py timeseries.h5 -d ${D0}_$D -g $G -o kite_ts
run save_gdal_ts      save_gdal.py timeseries.h5 -d $D -o ts.tif
run save_explorer_ts  save_explorer.py timeseries.h5 -v velocity.h5 -o exp
run save_qgis         save_qgis.py timeseries.h5 -g $G
run save_kmz_ts       save_kmz_timeseries.py timeseries.h5 -v -1 1 --noreference

#!/bin/bash
# Standalone tools left after the first scans. No package is downloaded: GDAL comes from the system.
SP=$1; S=$2; X=$3; code=$4; pp=$5; src=$6; label=$7
wd=$X/${label}_$code; mkdir -p $wd; cd $wd
cp -r $S/runs/$src/inputs . ; cp $S/runs/$src/*.h5 . 2>/dev/null; rm -f ERA5.h5; cp -r $S/runs/$src/geo . 2>/dev/null
export MPLBACKEND=Agg HOME=$SP/fakehome OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PATH=$SP/venv_np1/bin:$PATH GDAL_PAM_ENABLED=NO
unset PYTHONPATH; [ -n "$pp" ] && export PYTHONPATH=$pp
run() { n=$1; shift; timeout 300 "$@" > $n.log 2>&1; rc=$?; last=$(grep -E '^[A-Za-z_.]*(Error|Exception|Exit)|^[a-z_0-9.]*: error' $n.log | tail -1 | cut -c1-90); where=$(grep -E '^  File ".*mintpy' $n.log | tail -1 | sed -E 's#.*/mintpy/([^"]+)", line ([0-9]+).*#\1:\2#'); echo "$label $code $n rc=$rc $where $last"; }
I=inputs/ifgramStack.h5; G=$(ls inputs/geometry*.h5 | head -1)
GTS=$(ls geo/geo_timeseries*.h5 2>/dev/null | head -1); GV=geo/geo_velocity.h5; GG=$(ls geo/geo_geometry*.h5 2>/dev/null | head -1)
TS=$(ls timeseries*demErr.h5 timeseries*.h5 2>/dev/null | grep -v -E "Residual|ramp\.h5" | head -1)
D=$(info.py $TS --date | tail -1); D0=$(info.py $TS --date | head -1)
[ -z "$GTS" ] && { GTS=$TS; GV=velocity.h5; GG=$G; }
run kmz_ts       save_kmz_timeseries.py $GTS --vel $GV --tcoh $(ls geo/geo_temporalCoherence.h5 temporalCoherence.h5 2>/dev/null | head -1) --mask $(ls geo/geo_maskTempCoh.h5 maskTempCoh.h5 2>/dev/null | head -1) -o kmz_ts.kmz
run gmt_vel      save_gmt.py $GV -o v.grd
run gmt_ts       save_gmt.py $GTS $D -o t.grd
run gdal_vel     save_gdal.py $GV -o v.tif
run gdal_ts      save_gdal.py $GTS -d $D -o t.tif
run gdal_date12  save_gdal.py $GTS -d ${D0}_$D -o t12.tif
run qgis         save_qgis.py $GTS -g $GG -o q.gpkg
run stitch       bash -c "subset.py $GV -x 0 60 -o a.h5 && subset.py $GV -x 50 110 -o b.h5 && image_stitch.py a.h5 b.h5 -o s.h5 --nodisplay"
run asc_desc     bash -c "cp $GV d1.h5; cp $GV d2.h5; asc_desc2horz_vert.py d1.h5 d2.h5 -g $GG $GG --oo hz.h5"
run remove_dset  bash -c "cp $I st.h5; yes | remove_hdf5_dset.py st.h5 coherence"
run lookup       lookup_geo2radar.py $G -w lk.h5
run plate_motion plate_motion.py -g $G --om-cart 0.1 0.1 0.1 -v velocity.h5 --pmm-file pmm.h5
printf '%s\n' "$D0" > S1A_date.txt; printf '%s\n' "$D" > S1B_date.txt
run s1ab         s1ab_range_bias.py $TS -a compute -g $G
if [ "$label" = F1 ] || [ "$label" = F2 ]; then run iono_tec iono_tec.py $TS -g $G --tec-dir $SP/tec_data; fi

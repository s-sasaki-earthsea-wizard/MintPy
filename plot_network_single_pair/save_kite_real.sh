#!/usr/bin/env bash
# Run save_kite.py with the real kite package on one, two and three pair products.
# Usage: save_kite_real.sh <scratch root> <python of the kite venv>
# Products come from the p4final suite. save_kite only reads them, so main can run on them too.
SP=$1; PY=$2; S=$SP/sweep; R=$SP/kite_real
UP=$SP/src/MintPy-upstream/src; P4=$SP/src/patched_v9

run() {  # id code_root run_dir mode
  local id=$1 root=$2 d=$S/runs/p4final/$3 mode=$4 w=$R/$1 ts dates in dset rc
  mkdir -p "$w"
  ts=$(ls "$d"/geo/geo_timeseries*.h5 | head -1)
  dates=($("$PY" -I -c "import h5py,sys;f=h5py.File(sys.argv[1]);print(*[x.decode() for x in f['date'][()]])" "$ts"))
  if [ "$mode" = ts ]; then in=$ts; dset=${dates[0]}_${dates[-1]}; else in=$d/geo/geo_velocity.h5; dset=velocity; fi
  ( cd "$w" && HOME=$SP/fakehome MPLBACKEND=Agg PYTHONPATH=$root "$PY" -W ignore -m mintpy.cli.save_kite "$in" -d "$dset" \
      -g "$d/geo/geo_geometryRadar.h5" -m "$d/geo/geo_maskTempCoh.h5" -o kite_out > run.log 2>&1 ); rc=$?
  printf '%-10s rc=%s dset=%s\n' "$id" "$rc" "$dset"
  if [ "$rc" -eq 0 ]; then
    ( cd "$w" && "$PY" -W ignore "$R/check_container.py" kite_out "$in" "$dset" "$d/geo/geo_maskTempCoh.h5" "$d/geo/geo_geometryRadar.h5" 2>&1 | tail -1 )
  else
    grep -E "Error" "$w/run.log" | tail -1 | sed "s|$SP/||g"
  fi
}

for c in "up:$UP" "P4:$P4"; do
  for n in F1 F2 F3; do
    for m in ts vel; do run "${n}_${m}_${c%%:*}" "${c#*:}" "${n}_P4" "$m"; done
  done
done

echo '=== main and patched containers'
for n in F1 F2 F3; do
  for m in ts vel; do
    a=$R/${n}_${m}_up/kite_out; b=$R/${n}_${m}_P4/kite_out
    "$PY" -I -c "
import numpy as np, sys
a, b = [np.load(p + '.npz') for p in sys.argv[2:4]]
same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
print(sys.argv[1], 'npz_identical=%s' % same, end=' ')" "${n}_${m}" "$a" "$b"
    cmp -s "$a.yml" "$b.yml" && echo 'yml_identical=True' || echo 'yml_identical=False'
  done
done

"""A PASS only counts if the option-specific product exists and the log shows the step actually ran."""
import glob
import os
import re
import sys

R = sys.argv[1]
EXPECT = {
    'baseline': (['timeseries.h5', 'velocity.h5'], None),
    'unwrap_bridging': (['timeseries.h5', 'velocity.h5'], r'unwrap_error_bridging\.py'),
    'weight_var': (['timeseries.h5'], r'-w var|weightFunc.*var'),
    'weight_fim': (['timeseries.h5'], r'-w fim|weightFunc.*fim'),
    'weight_coh': (['timeseries.h5'], r'-w coh|weightFunc.*coh'),
    'LOD_Env': (['timeseries_LOD.h5'], r'local_oscilator_drift\.py'),
    'SET': (['timeseries_SET.h5', 'inputs/SET.h5'], r'solid_earth_tides\.py'),
    'iono_split': (['timeseries_ion.h5', 'inputs/ion.h5'], r'iono_split_spectrum\.py'),
    'tropo_pyaps': (['timeseries_ERA5.h5', 'velocityERA5.h5'], r'tropo_pyaps3\.py'),
    'tropo_height': (['timeseries_tropHgt.h5'], r'tropo_phase_elevation\.py'),
    'deramp_linear': (['timeseries_ramp.h5'], r'remove_ramp\.py'),
    'deramp_quadratic': (['timeseries_ramp.h5'], r'remove_ramp\.py'),
    'topo_default': (['timeseries_demErr.h5', 'demErr.h5', 'timeseriesResidual.h5'], r'dem_error\.py'),
    'topo_poly1': (['timeseries_demErr.h5', 'demErr.h5'], r'dem_error\.py'),
    'topo_pixelwise': (['timeseries_demErr.h5', 'demErr.h5'], r'dem_error\.py'),
    'topo_phasevel': (['timeseries_demErr.h5', 'demErr.h5'], r'dem_error\.py'),
    'rms_deramp_linear': (['rms_timeseriesResidual_ramp.txt'], r'timeseries_rms\.py'),
    'rms_deramp_no': (['rms_timeseriesResidual.txt'], r'timeseries_rms\.py'),
    'refdate_first': (['velocity.h5'], r'reference_date\.py'),
    'refdate_second': (['velocity.h5'], r'reference_date\.py'),
    'geocode': (['geo/geo_velocity.h5', 'geo/geo_timeseries*.h5'], r'geocode\.py'),
    'kmz': (['geo/*.kmz'], r'save_kmz\.py'),
    'hdfeos5': (['*.he5'], r'save_hdfeos5\.py'),
    'plot': (['pic/velocity.png', 'pic/network.pdf'], r'plot_smallbaselineApp'),
    'net_coh_mst': (['velocity.h5'], r'coherence-based network modification'),
    'net_area': (['velocity.h5'], r'area-ratio-based network modification'),
}
miss = ok = 0
for d in sorted(glob.glob(f'{R}/*1_*')):
    case = os.path.basename(d)
    tag = case.split('_', 1)[1]
    if tag not in EXPECT:
        continue
    log = open(f'{d}/out.log').read()
    if 'Normal end' not in log:
        continue
    files, pat = EXPECT[tag]
    lacking = [f for f in files if not glob.glob(f'{d}/{f}')]
    no_log = pat and not re.search(pat, log)
    if lacking or no_log:
        miss += 1
        print(f'CHECK {case}: missing={lacking} log_match={not no_log}; present: {sorted(os.path.basename(x) for x in glob.glob(d + "/*.h5"))[:14]}')
    else:
        ok += 1
print(f'artifact check: ok={ok} problem={miss}')

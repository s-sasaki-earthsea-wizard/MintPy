NAMES = {'F': 'FernandinaSenDT128', 'S': 'SanFranSenDT42', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
B0 = {
    'mintpy.compute.cluster': 'no', 'mintpy.unwrapError.method': 'no', 'mintpy.networkInversion.weightFunc': 'no',
    'mintpy.solidEarthTides': 'no', 'mintpy.ionosphericDelay.method': 'no', 'mintpy.troposphericDelay.method': 'no',
    'mintpy.deramp': 'no', 'mintpy.topographicResidual': 'no', 'mintpy.geocode': 'no', 'mintpy.save.kmz': 'no',
    'mintpy.save.hdfEos5': 'no', 'mintpy.plot': 'no',
}
CASES = []
def add(tag, ds_list, cfg=None, setup=None, code='P3'):
    for ds in ds_list:
        CASES.append(dict(id=f'{ds}1_{tag}', base=f'{ds}1', code=code, official=NAMES[ds],
                          cfg={**B0, **(cfg or {})}, setup=setup))

ION = 'cp inputs/ifgramStack.h5 inputs/ionStack.h5'
ENV = "python -c \"import h5py,glob;[h5py.File(p,'r+').attrs.__setitem__('PLATFORM','Env') for p in glob.glob('inputs/*.h5')]\""
add('baseline', 'FSKB')
for m in ('bridging', 'phase_closure', 'bridging+phase_closure'):
    add(f'unwrap_{m}', 'FS', {'mintpy.unwrapError.method': m})
for m in ('var', 'fim', 'coh'):
    add(f'weight_{m}', 'FB', {'mintpy.networkInversion.weightFunc': m})
add('LOD_Env', 'F', setup=ENV)
add('SET', 'FB', {'mintpy.solidEarthTides': 'yes'})
add('iono_split', 'FB', {'mintpy.ionosphericDelay.method': 'split_spectrum'}, setup=ION)
add('tropo_pyaps', 'FK', {'mintpy.troposphericDelay.method': 'pyaps', 'mintpy.troposphericDelay.weatherModel': 'ERA5'})
add('tropo_height', 'FB', {'mintpy.troposphericDelay.method': 'height_correlation'})
add('tropo_gacos_nodir', 'F', {'mintpy.troposphericDelay.method': 'gacos'})
for m in ('linear', 'quadratic'):
    add(f'deramp_{m}', 'FB', {'mintpy.deramp': m})
add('topo_default', 'FB', {'mintpy.topographicResidual': 'yes'})
add('topo_poly1', 'FB', {'mintpy.topographicResidual': 'yes', 'mintpy.topographicResidual.polyOrder': '1'})
add('topo_pixelwise', 'FB', {'mintpy.topographicResidual': 'yes', 'mintpy.topographicResidual.pixelwiseGeometry': 'yes'})
add('topo_phasevel', 'FB', {'mintpy.topographicResidual': 'yes', 'mintpy.topographicResidual.phaseVelocity': 'yes'})
for m in ('linear', 'no'):
    add(f'rms_deramp_{m}', 'F', {'mintpy.topographicResidual': 'yes', 'mintpy.residualRMS.deramp': m})
add('refdate_no', 'F', {'mintpy.reference.date': 'no'})
add('refdate_first', 'F', {'mintpy.reference.date': '20180327'})
add('refdate_second', 'F', {'mintpy.reference.date': '20180408'})
add('vel_bootstrap', 'F', {'mintpy.timeFunc.uncertaintyQuantification': 'bootstrap', 'mintpy.timeFunc.bootstrapCount': '50'})
add('vel_poly2', 'F', {'mintpy.timeFunc.polynomial': '2'})
add('geocode', 'FK', {'mintpy.geocode': 'yes'})
add('kmz', 'FK', {'mintpy.geocode': 'yes', 'mintpy.save.kmz': 'yes'})
add('hdfeos5', 'FB', {'mintpy.save.hdfEos5': 'yes'})
add('plot', 'FKB', {'mintpy.plot': 'yes'})
add('net_coh_mst', 'F', {'mintpy.network.coherenceBased': 'yes', 'mintpy.network.minCoherence': '0.99', 'mintpy.network.keepMinSpanTree': 'yes'})
add('net_coh_nomst', 'F', {'mintpy.network.coherenceBased': 'yes', 'mintpy.network.minCoherence': '0.99', 'mintpy.network.keepMinSpanTree': 'no'})
add('net_area', 'F', {'mintpy.network.areaRatioBased': 'yes', 'mintpy.network.minAreaRatio': '0.5'})
add('net_tbase1', 'F', {'mintpy.network.tempBaseMax': '1'})
add('net_exclude_all', 'F', {'mintpy.network.excludeIfgIndex': '0'})
# stack with several rows but a single kept interferogram, reached with a stock option
for ds, d1, d2 in (('K', '20090829', '20091014'), ('B', '20191024', '20191030'), ('F', '20161120', '20161202'), ('S', '20180824', '20180905')):
    for code in ('up', 'P1', 'P2', 'P3'):
        cfg = {'mintpy.network.startDate': d1, 'mintpy.network.endDate': d2}
        if ds == 'F':
            cfg['mintpy.unwrapError.method'] = 'no'
        if ds == 'S':
            cfg['mintpy.troposphericDelay.method'] = 'no'
        CASES.append(dict(id=f'{ds}5_keep1_{code}', base=f'{ds}5', code=code, official=NAMES[ds], cfg=cfg))
# official configs with the two environment-only blockers lifted
for ds, cfg in (('F', {'mintpy.unwrapError.method': 'no'}), ('S', {'mintpy.troposphericDelay.method': 'no'})):
    for n, code in ((1, 'up'), (1, 'P1'), (1, 'P2'), (1, 'P3'), (2, 'up'), (3, 'up'), (5, 'up'), (2, 'P3')):
        CASES.append(dict(id=f'{ds}{n}_off_{code}', base=f'{ds}{n}', code=code, official=NAMES[ds], cfg=cfg))

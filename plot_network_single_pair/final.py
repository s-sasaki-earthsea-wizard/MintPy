NAMES = {'F': 'FernandinaSenDT128'}
B0 = {
    'mintpy.compute.cluster': 'no', 'mintpy.unwrapError.method': 'no', 'mintpy.networkInversion.weightFunc': 'no',
    'mintpy.solidEarthTides': 'no', 'mintpy.ionosphericDelay.method': 'no', 'mintpy.troposphericDelay.method': 'no',
    'mintpy.deramp': 'no', 'mintpy.topographicResidual': 'no', 'mintpy.geocode': 'no', 'mintpy.save.kmz': 'no',
    'mintpy.save.hdfEos5': 'no', 'mintpy.plot': 'no',
}
CASES = []
RAW = f'{SP}/sweep/rawK'
for n, code in ((1, 'up'), (1, 'P3'), (2, 'up'), (2, 'P3')):
    CASES.append(dict(id=f'raw{n}_nogeo_{code}', base='empty', code=code,
                      cfg={'mintpy.troposphericDelay.method': 'no', 'mintpy.compute.cluster': 'no', 'mintpy.geocode': 'no',
                           'mintpy.save.kmz': 'no'},
                      setup=f'cat {RAW}/n{n}/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))
CASES.append(dict(id='F2_poly2_P3', base='F2', code='P3', official=NAMES['F'], cfg={**B0, 'mintpy.timeFunc.polynomial': '2'}))
GA = {'mintpy.troposphericDelay.method': 'gacos'}
CASES.append(dict(id='F1_gacos_P3', base='F1', code='P3', official=NAMES['F'], cfg={**B0, **GA}, setup=f'python {SP}/sweep/make_gacos.py'))
CASES.append(dict(id='F2_gacos_up', base='F2', code='up', official=NAMES['F'], cfg={**B0, **GA}, setup=f'python {SP}/sweep/make_gacos.py'))
PN = 'plot_network.py ifgramStack.h5 --nodisplay'
for code in ('up', 'P3'):
    CASES += [
        dict(id=f'syn_plot_first_{code}', base='syn1', code=code, cmd=PN),
        dict(id=f'syn_plot_repeat_{code}', base='syn1', code=code, cmd=f'{PN} ; {PN}'),
        dict(id=f'syn_plot_txt_{code}', base='syn1', code=code, cmd='plot_network.py one.txt --nodisplay'),
        dict(id=f'syn_modnet_{code}', base='syn1', code=code, cmd='modify_network.py ifgramStack.h5 --coherence-based --min-coherence 0.5'),
        dict(id=f'syn_spatial_{code}', base='syn1', code=code, cmd='spatial_average.py ifgramStack.h5 -d coherence --nodisplay'),
        dict(id=f'syn2_plot_{code}', base='syn2', code=code, cmd=PN),
        dict(id=f'syn2_plot_txt_{code}', base='syn2', code=code, cmd='plot_network.py two.txt --nodisplay'),
    ]

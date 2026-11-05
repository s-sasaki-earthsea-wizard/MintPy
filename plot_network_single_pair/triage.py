NAMES = {'F': 'FernandinaSenDT128', 'S': 'SanFranSenDT42', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
B0 = {
    'mintpy.compute.cluster': 'no', 'mintpy.unwrapError.method': 'no', 'mintpy.networkInversion.weightFunc': 'no',
    'mintpy.solidEarthTides': 'no', 'mintpy.ionosphericDelay.method': 'no', 'mintpy.troposphericDelay.method': 'no',
    'mintpy.deramp': 'no', 'mintpy.topographicResidual': 'no', 'mintpy.geocode': 'no', 'mintpy.save.kmz': 'no',
    'mintpy.save.hdfEos5': 'no', 'mintpy.plot': 'no',
}
CASES = []
def add(tag, ds, n, cfg, code='up'):
    CASES.append(dict(id=f'{ds}{n}_{tag}_{code}', base=f'{ds}{n}', code=code, official=NAMES[ds], cfg={**B0, **cfg}))
# is each non-crash-chain failure specific to one pair?
for n in (2, 3, 5):
    add('bootstrap', 'F', n, {'mintpy.timeFunc.uncertaintyQuantification': 'bootstrap', 'mintpy.timeFunc.bootstrapCount': '50'})
    add('poly2', 'F', n, {'mintpy.timeFunc.polynomial': '2'})
    add('gacos', 'F', n, {'mintpy.troposphericDelay.method': 'gacos'})
    add('hdfeos5', 'B', n, {'mintpy.save.hdfEos5': 'yes'})
# non-regression of the three experimental changes on the complete official run
CASES.append(dict(id='Ffull_P3', base='Ffull', code='P3', official='FernandinaSenDT128', timeout=2400))
# raw ROI_PAC products -> whole workflow
RAW = f'{SP}/sweep/rawK'
for n, code in ((1, 'up'), (1, 'P3'), (2, 'up'), (2, 'P3')):
    CASES.append(dict(id=f'raw{n}_{code}', base='empty', code=code,
                      cfg={'mintpy.troposphericDelay.method': 'no', 'mintpy.compute.cluster': 'no'},
                      setup=f'cat {RAW}/n{n}/load.txt >> case.txt',
                      cmd='smallbaselineApp.py case.txt --start load_data'))

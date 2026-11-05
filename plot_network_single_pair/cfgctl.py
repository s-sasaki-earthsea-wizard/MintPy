NAMES = {'F': 'FernandinaSenDT128'}
B0 = {"mintpy.compute.cluster": "no", "mintpy.unwrapError.method": "no", "mintpy.networkInversion.weightFunc": "no", "mintpy.solidEarthTides": "no", "mintpy.ionosphericDelay.method": "no", "mintpy.troposphericDelay.method": "no", "mintpy.deramp": "no", "mintpy.topographicResidual": "no", "mintpy.geocode": "no", "mintpy.save.kmz": "no", "mintpy.save.hdfEos5": "no", "mintpy.plot": "no"}
CASES = []
def add(tag, cfg, base='F2', code='up'):
    CASES.append(dict(id=f'{base}_{tag}_{code}', base=base, code=code, official=NAMES['F'], cfg={**B0, **cfg}))
add('minRedundancy2', {'mintpy.networkInversion.minRedundancy': '2'})
add('minNumPixel', {'mintpy.networkInversion.minNumPixel': '100000000'})
add('deramp_mask_no', {'mintpy.deramp': 'linear', 'mintpy.deramp.maskFile': 'no'})
add('poly0', {'mintpy.timeFunc.polynomial': '0'})
add('tf_excl_to_one', {'mintpy.timeFunc.excludeDate': '20170107,20170119'})
add('periodic', {'mintpy.timeFunc.periodic': '1'})
add('step', {'mintpy.timeFunc.stepDate': '20170110'})
add('exp', {'mintpy.timeFunc.exp': '20170110,60'})
add('log', {'mintpy.timeFunc.log': '20170110,60'})
add('geo_linear', {'mintpy.geocode': 'yes', 'mintpy.geocode.interpMethod': 'linear'})
add('periodic', {'mintpy.timeFunc.periodic': '1'}, base='F3')
add('periodic', {'mintpy.timeFunc.periodic': '1'}, base='F3', code='P4')

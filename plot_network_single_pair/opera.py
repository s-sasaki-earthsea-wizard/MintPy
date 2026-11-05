NAMES = {'F': 'FernandinaSenDT128', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42', 'Sx': 'SanFranSenDT42'}
B0 = {"mintpy.compute.cluster": "no", "mintpy.unwrapError.method": "no", "mintpy.networkInversion.weightFunc": "no", "mintpy.solidEarthTides": "no", "mintpy.ionosphericDelay.method": "no", "mintpy.troposphericDelay.method": "no", "mintpy.deramp": "no", "mintpy.topographicResidual": "no", "mintpy.geocode": "no", "mintpy.save.kmz": "no", "mintpy.save.hdfEos5": "no", "mintpy.plot": "no"}
WORKERS = 2
CASES = []
for ds, n, code in (('F', 1, 'P4'), ('F', 2, 'up'), ('F', 2, 'P4'), ('B', 1, 'P4'), ('B', 2, 'up'), ('Sx', 1, 'P4'), ('Sx', 2, 'up')):
    CASES.append(dict(id=f'{ds}{n}_opera_{code}', base=f'{ds}{n}', code=code, official=NAMES[ds], timeout=1800,
                      cfg={**B0, 'mintpy.troposphericDelay.method': 'opera', 'mintpy.troposphericDelay.operaDir': f'{SP}/opera_data/{ds}'}))

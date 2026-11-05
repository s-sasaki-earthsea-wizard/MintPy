NAMES = {'F': 'FernandinaSenDT128', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42', 'Sx': 'SanFranSenDT42'}
B0 = {"mintpy.compute.cluster": "no", "mintpy.unwrapError.method": "no", "mintpy.networkInversion.weightFunc": "no", "mintpy.solidEarthTides": "no", "mintpy.ionosphericDelay.method": "no", "mintpy.troposphericDelay.method": "no", "mintpy.deramp": "no", "mintpy.topographicResidual": "no", "mintpy.geocode": "no", "mintpy.save.kmz": "no", "mintpy.save.hdfEos5": "no", "mintpy.plot": "no"}
CASES = []
for ds, n, code, w in (('F', 1, 'P4', 'no'), ('F', 1, 'P4', 'var'), ('K', 1, 'P4', 'no'), ('B', 1, 'P4', 'no'), ('Sx', 1, 'P4', 'no'), ('F', 2, 'up', 'no'), ('F', 2, 'P4', 'no'), ('K', 2, 'up', 'no'), ('B', 2, 'up', 'no')):
    cmd = ('smallbaselineApp.py case.txt --start modify_network --end reference_point'
           ' && info.py inputs/ifgramStack.h5 --date | head -1 | cut -d_ -f1 > reference_date.txt'
           f' && ifgram_inversion.py inputs/ifgramStack.h5 -w {w} --calc-cov -o tsx.h5 tcohx.h5 numx.h5'
           ' && smallbaselineApp.py case.txt --start quick_overview')
    CASES.append(dict(id=f'{ds}{n}_cov_{w}_{code}', base=f'{ds}{n}', code=code, official=NAMES[ds], cmd=cmd,
                      cfg={**B0, 'mintpy.timeFunc.uncertaintyQuantification': 'covariance', 'mintpy.timeFunc.timeSeriesCovFile': 'tsxDecorCov.h5'}))

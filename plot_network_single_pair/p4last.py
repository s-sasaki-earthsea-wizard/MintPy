NAMES = {'F': 'FernandinaSenDT128', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
B0 = {'mintpy.compute.cluster': 'no', 'mintpy.unwrapError.method': 'no', 'mintpy.networkInversion.weightFunc': 'no',
      'mintpy.solidEarthTides': 'no', 'mintpy.ionosphericDelay.method': 'no', 'mintpy.troposphericDelay.method': 'no',
      'mintpy.deramp': 'no', 'mintpy.topographicResidual': 'no', 'mintpy.geocode': 'no', 'mintpy.save.kmz': 'no',
      'mintpy.save.hdfEos5': 'no'}
CASES = [dict(id=f'{d}1_plot_P4', base=f'{d}1', code='P4', official=NAMES[d], cfg={**B0, 'mintpy.plot': 'yes'}) for d in 'FKB']
RAW = f'{SP}/sweep/rawK'
CASES.append(dict(id='raw1_nogeo_P4', base='empty', code='P4',
                  cfg={'mintpy.troposphericDelay.method': 'no', 'mintpy.compute.cluster': 'no', 'mintpy.geocode': 'no', 'mintpy.save.kmz': 'no'},
                  setup=f'cat {RAW}/n1/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))

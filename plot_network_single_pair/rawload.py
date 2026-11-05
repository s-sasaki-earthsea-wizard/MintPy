VENV = f'{SP}/venv_np1'
WORKERS = 4
NAMES = {'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42', 'A': 'SanFranSenDT42', 'F': 'FernandinaSenDT128'}
CFG = {'mintpy.compute.cluster': 'no', 'mintpy.troposphericDelay.method': 'no', 'mintpy.unwrapError.method': 'no'}
CASES = []
for ds in 'KBAF':
    for n, code in ((1, 'upG'), (1, 'P4G'), (2, 'upG'), (2, 'P4G')):
        CASES.append(dict(id=f'raw{ds}{n}_{code}', base='empty', code=code, official=NAMES[ds], cfg=CFG, timeout=1200,
                          setup=f'cat {SP}/rawone/{ds}{n}/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))
# load-time options on a real one-pair ROI_PAC tree
for tag, extra in (('subset', {'mintpy.subset.yx': '0:400,0:400', 'mintpy.geocode': 'no'}), ('multilook', {'mintpy.multilook.ystep': '2', 'mintpy.multilook.xstep': '2', 'mintpy.geocode': 'no'}),
                   ('gzip', {'mintpy.load.compression': 'gzip'})):
    CASES.append(dict(id=f'rawK1_{tag}_P4G', base='empty', code='P4G', official=NAMES['K'], cfg={**CFG, **extra}, timeout=900,
                      setup=f'cat {SP}/rawone/K1/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))

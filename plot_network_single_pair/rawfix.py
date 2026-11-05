VENV = f'{SP}/venv_np1'
WORKERS = 4
NAMES = {'K': 'KujuAlosAT422F650', 'F': 'FernandinaSenDT128'}
CFG = {'mintpy.compute.cluster': 'no', 'mintpy.troposphericDelay.method': 'no', 'mintpy.unwrapError.method': 'no'}
CASES = []
for n, code in ((1, 'upG'), (1, 'P4G'), (2, 'upG'), (2, 'P4G')):
    CASES.append(dict(id=f'rawF{n}_{code}', base='empty', code=code, official=NAMES['F'], cfg=CFG, timeout=1200,
                      setup=f'cat {SP}/rawone/F{n}/load.txt >> case.txt',
                      cmd='cp case.txt FernandinaSenDT128.txt && smallbaselineApp.py FernandinaSenDT128.txt --start load_data'))
SUB = {'mintpy.subset.yx': '1200:1400,100:300', 'mintpy.geocode': 'no'}
for n, code in ((1, 'P4G'), (2, 'upG')):
    CASES.append(dict(id=f'rawK{n}_subset_{code}', base='empty', code=code, official=NAMES['K'], cfg={**CFG, **SUB}, timeout=900,
                      setup=f'cat {SP}/rawone/K{n}/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))

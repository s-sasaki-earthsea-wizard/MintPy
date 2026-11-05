VENV = f'{SP}/venv_np1'
WORKERS = 4
NAMES = {'A': 'SanFranSenDT42', 'F': 'FernandinaSenDT128'}
CFG = {'mintpy.compute.cluster': 'no', 'mintpy.troposphericDelay.method': 'no', 'mintpy.unwrapError.method': 'no', 'mintpy.geocode': 'no', 'mintpy.reference.lalo': 'auto', 'mintpy.reference.yx': 'auto'}
CASES = []
for n, code in ((1, 'P4G'), (2, 'upG')):
    CASES.append(dict(id=f'rawF{n}_subset_{code}', base='empty', code=code, official=NAMES['F'], cfg={**CFG, 'mintpy.subset.yx': '100:300,100:300'}, timeout=900,
                      setup=f'cat {SP}/rawone/F{n}/load.txt >> case.txt', cmd='cp case.txt FernandinaSenDT128.txt && smallbaselineApp.py FernandinaSenDT128.txt --start load_data'))
    CASES.append(dict(id=f'rawA{n}_subset_{code}', base='empty', code=code, official=NAMES['A'], cfg={**CFG, 'mintpy.subset.lalo': '37.5:37.8,-122.5:-122.2'}, timeout=900,
                      setup=f'cat {SP}/rawone/A{n}/load.txt >> case.txt', cmd='smallbaselineApp.py case.txt --start load_data'))
for n, code in ((1, 'up'), (1, 'P4'), (2, 'up'), (2, 'P4')):
    CASES.append(dict(id=f'offset{n}_{code}', base='empty', code=code, setup=f'cp {SP}/offset_n{n}.h5 offsetStack.h5',
                      cmd='ifgram_inversion.py offsetStack.h5 -i rangeOffset -w no -o ts.h5 tcoh.h5 num.h5 --update'))

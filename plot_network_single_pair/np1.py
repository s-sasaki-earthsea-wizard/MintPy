VENV = f'{SP}/venv_np1'
NAMES = {'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
CASES = []
for ds in 'KB':
    for n, code in ((1, 'up'), (1, 'P1'), (1, 'P2'), (1, 'P3'), (2, 'up'), (3, 'up'), (2, 'P3')):
        CASES.append(dict(id=f'{ds}{n}_{code}', base=f'{ds}{n}', code=code, official=NAMES[ds]))

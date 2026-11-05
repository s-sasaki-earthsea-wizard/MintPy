VENV = f'{SP}/venv_np1'
NAMES = {'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
CASES = []
for ds in 'KB':
    for n in (1, 2):
        CASES.append(dict(id=f'{ds}{n}_P4', base=f'{ds}{n}', code='P4', official=NAMES[ds]))

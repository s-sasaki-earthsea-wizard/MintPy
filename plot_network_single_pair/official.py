DATASETS = {'F': 'FernandinaSenDT128', 'S': 'SanFranSenDT42', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42'}
CASES = []
for key, name in DATASETS.items():
    for n in (1, 2, 3, 5):
        CASES.append(dict(id=f'{key}{n}_up', base=f'{key}{n}', code='up', official=name))
    for code in ('P1', 'P2', 'P3'):
        CASES.append(dict(id=f'{key}1_{code}', base=f'{key}1', code=code, official=name))
    for n in (2, 3):
        CASES.append(dict(id=f'{key}{n}_P3', base=f'{key}{n}', code='P3', official=name))

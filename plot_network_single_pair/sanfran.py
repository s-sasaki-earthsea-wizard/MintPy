CFG = {'mintpy.troposphericDelay.method': 'no'}
NAME = 'SanFranSenDT42'
CASES = []
for n, code in ((1, 'up'), (1, 'P1'), (1, 'P2'), (1, 'P3'), (2, 'up'), (3, 'up'), (5, 'up'), (2, 'P3'), (3, 'P3')):
    CASES.append(dict(id=f'Sx{n}_{code}', base=f'Sx{n}', code=code, official=NAME, cfg=CFG))
for code in ('up', 'P1', 'P2', 'P3'):
    CASES.append(dict(id=f'Sx5_keep1_{code}', base='Sx5', code=code, official=NAME,
                      cfg={**CFG, 'mintpy.network.startDate': '20180824', 'mintpy.network.endDate': '20180905'}))

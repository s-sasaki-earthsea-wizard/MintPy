NAMES = {'F': 'FernandinaSenDT128', 'K': 'KujuAlosAT422F650', 'B': 'SanFranBaySenD42', 'S': 'SanFranSenDT42'}
XF = {'mintpy.unwrapError.method': 'no'}
XS = {'mintpy.troposphericDelay.method': 'no'}
CASES = []
for ds, base, cfg in (('F', 'F', XF), ('K', 'K', {}), ('B', 'B', {}), ('S', 'Sx', XS)):
    for n in (1, 2, 3):
        CASES.append(dict(id=f'{base}{n}_P4', base=f'{base}{n}', code='P4', official=NAMES[ds], cfg=cfg))
for ds, base, cfg, d1, d2 in (('K', 'K', {}, '20090829', '20091014'), ('B', 'B', {}, '20191024', '20191030'),
                              ('F', 'F', XF, '20161120', '20161202'), ('S', 'Sx', XS, '20180824', '20180905')):
    CASES.append(dict(id=f'{base}5_keep1_P4', base=f'{base}5', code='P4', official=NAMES[ds],
                      cfg={**cfg, 'mintpy.network.startDate': d1, 'mintpy.network.endDate': d2}))
CASES.append(dict(id='Ffull_P4', base='Ffull', code='P4', official=NAMES['F']))

CASES = []
for n, code in ((1, 'up'), (1, 'P4'), (2, 'up'), (2, 'P4')):
    CASES.append(dict(id=f'offset{n}_{code}', base='empty', code=code, setup=f'cp {SP}/offset_n{n}.h5 offsetStack.h5',
                      cmd='ifgram_inversion.py offsetStack.h5 -i rangeOffset -w no -o ts.h5 tcoh.h5 num.h5 --update'))

"""Run every entry point against every code variant and print a compact result table."""
import os
import re
import shutil
import subprocess
import sys

sp = sys.argv[1]
venv_bin = f'{sp}/venv/bin'
variants = {
    'V6 final': f'{sp}/src/patched_v6',
}
cases = {
    'plot_network 1st':  ('stack1', ['plot_network.py ifgramStack.h5 --nodisplay']),
    'plot_network 2nd':  ('stack1', ['plot_network.py ifgramStack.h5 --nodisplay', 'plot_network.py ifgramStack.h5 --nodisplay']),
    'plot_network 2pair': ('stack2', ['plot_network.py ifgramStack.h5 --nodisplay']),
    'modify_net coh':    ('stack1', ['modify_network.py ifgramStack.h5 --coherence-based --min-coherence 0.5']),
    'modify_net coh low': ('stack1', ['modify_network.py ifgramStack.h5 --coherence-based --min-coherence 0.95']),
    'spatial_avg stack': ('stack1', ['spatial_average.py ifgramStack.h5 -d coherence --nodisplay']),
    'spatial_avg ts1':   ('other', ['spatial_average.py timeseries1.h5 --nodisplay']),
    'spatial_avg vel':   ('other', ['spatial_average.py velocity.h5 --nodisplay']),
    'sbApp modify_net':  ('geo', ['smallbaselineApp.py repro.txt --dostep modify_network']),
}
env0 = dict(os.environ, MPLBACKEND='Agg', PATH=f'{venv_bin}:' + os.environ['PATH'])
env0.pop('VIRTUAL_ENV', None)
py = f'{venv_bin}/python'
for vname, pp in variants.items():
    print(f'\n===== {vname}')
    env = dict(env0)
    if pp:
        env['PYTHONPATH'] = pp
    for cname, (setup, cmds) in cases.items():
        wd = f'{sp}/repro/work/m/{vname}/{cname}'.replace(' ', '_')
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        if setup == 'stack1':
            subprocess.run([py, f'{sp}/repro/make_stack.py', '1'], cwd=wd, check=True)
        elif setup == 'stack2':
            subprocess.run([py, f'{sp}/repro/make_stack.py', '2'], cwd=wd, check=True)
        elif setup == 'other':
            subprocess.run([py, f'{sp}/repro/make_other.py'], cwd=wd, check=True)
        elif setup == 'geo':
            subprocess.run([py, f'{sp}/repro/make_geo_case.py'], cwd=wd, check=True)
            open(f'{wd}/repro.txt', 'w').write('mintpy.load.processor = nisar\n')
        for c in cmds:
            r = subprocess.run(c, shell=True, cwd=wd, env=env, capture_output=True, text=True)
            open(f'{wd}/out.log', 'a').write(r.stdout + r.stderr)
        tail = (r.stderr.strip().split('\n') or [''])[-1]
        last_frame = re.findall(r'File "[^"]*/mintpy/([^"]+)", line (\d+)', r.stderr)
        where = f' @ {last_frame[-1][0]}:{last_frame[-1][1]}' if (r.returncode and last_frame) else ''
        status = 'OK  ' if r.returncode == 0 else 'FAIL'
        print(f'{status} {cname:20s} {tail[:95] if r.returncode else ""}{where}')

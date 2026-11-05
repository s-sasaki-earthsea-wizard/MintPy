"""Run smallbaselineApp cases in parallel and tabulate how far each one got."""
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time

SP = sys.argv[1]
SUITE = sys.argv[2]
exec(open(f'{SP}/sweep/{SUITE}.py').read())  # defines CASES

CODE = {
    'up': None,
    'P1': f'{SP}/src/patched_v6',
    'P2': f'{SP}/src/patched_v7',
    'P3': f'{SP}/src/patched_v8',
}
for extra in sorted(os.listdir(f'{SP}/src')):
    if extra.startswith('exp_'):
        CODE[extra[4:]] = f'{SP}/src/{extra}'
BASE = f'{SP}/sweep/bases'
VENV = globals().get('VENV', f'{SP}/venv')
OUT = f'{SP}/sweep/runs/{SUITE}'
UPCFG = f'{SP}/src/MintPy-upstream/tests/configs'


def run_case(c):
    wd = f"{OUT}/{c['id']}"
    shutil.rmtree(wd, ignore_errors=True)
    shutil.copytree(f"{BASE}/{c['base']}", wd, symlinks=False)
    lines = []
    if c.get('official'):
        lines.append(open(f"{UPCFG}/{c['official']}.txt").read())
    lines += [f'{k} = {v}' for k, v in c.get('cfg', {}).items()]
    open(f'{wd}/case.txt', 'w').write('\n'.join(lines) + '\n')
    if c.get('setup'):
        subprocess.run(c['setup'], shell=True, cwd=wd, check=True, env=dict(os.environ, PATH=f'{VENV}/bin:' + os.environ['PATH']))
    env = dict(os.environ, MPLBACKEND='Agg', HOME=f'{SP}/fakehome', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PATH=f'{VENV}/bin:' + os.environ['PATH'])
    env.pop('VIRTUAL_ENV', None)
    env.pop('PYTHONPATH', None)
    if CODE[c['code']]:
        env['PYTHONPATH'] = CODE[c['code']]
    cmd = c.get('cmd', 'smallbaselineApp.py case.txt --start modify_network')
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, cwd=wd, env=env, capture_output=True, text=True, timeout=c.get('timeout', 900))
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = 124, (e.stdout or b'').decode(errors='replace'), (e.stderr or b'').decode(errors='replace')
    log = so + '\n' + se
    open(f'{wd}/out.log', 'w').write(log)
    steps = re.findall(r'^\*+ step - (\w+) \*+', log, flags=re.M)
    tail = [ln for ln in se.strip().split('\n') if ln.strip()]
    frames = re.findall(r'File "[^"]*/mintpy/([^"]+)", line (\d+)', se)
    site = f'{frames[-1][0]}:{frames[-1][1]}' if rc and frames else ''
    res = dict(id=c['id'], code=c['code'], rc=rc, steps=steps, last=steps[-1] if steps else '', site=site,
               err=(tail[-1][:150] if rc and tail else ''), normal='Normal end' in log,
               tracebacks=log.count('Traceback (most recent call last)'), secs=round(time.time() - t0),
               warns=len(re.findall(r'RuntimeWarning', log)))
    for ln in reversed(tail):  # keep the exception, not trailing advice lines
        if re.match(r'^[\w.]*(Error|Exception)\b', ln):
            res['err'] = ln[:150]
            break
    return res


with cf.ThreadPoolExecutor(max_workers=globals().get('WORKERS', 6)) as ex:
    results = list(ex.map(run_case, CASES))
with open(f'{SP}/sweep/{SUITE}.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
for r in results:
    flag = 'PASS' if r['rc'] == 0 else ('TIME' if r['rc'] == 124 else 'FAIL')
    print(f"{flag} {r['id']:<34} {r['code']:<4} {r['secs']:>4}s last={r['last']:<20} tb={r['tracebacks']} {r['site']} {r['err']}")

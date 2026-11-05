"""Build a one- or two-pair raw tree without touching the source dataset.

Metadata that prep_* may rewrite is copied. Binary rasters are symlinked and only read.
"""
import datetime as dt
import glob
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

SP, KEY, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
BENCH = os.path.expanduser('~/MintPy_bench')
SRC = {'K': f'{BENCH}/KujuAlosAT422F650/KujuAlosAT422F650', 'B': f'{BENCH}/SanFranBaySenD42/SanFranBaySenD42',
       'A': f'{BENCH}/SanFranSenDT42/SanFranSenDT42', 'F': f'{BENCH}/FernandinaSenDT128'}[KEY]
OUT = f'{SP}/rawone/{KEY}{N}'
META = ('.rsc', '.xml', '.vrt', '.hdr', '.txt', '.dat', '.PRM', '.csh', '.png')


def parse(name):
    d1, d2 = name.split('_')[:2]
    if len(d1) == 7:  # GMTSAR tag: year + zero-based day of year
        return tuple(dt.datetime(int(d[:4]), 1, 1) + dt.timedelta(days=int(d[4:])) for d in (d1, d2))
    return dt.datetime.strptime(d1, '%Y%m%d'), dt.datetime.strptime(d2, '%Y%m%d')


def chain(names, n):
    """Return a chain of n connected pairs, shortest spans first."""
    pairs = sorted(names, key=lambda x: (parse(x)[1] - parse(x)[0], x))
    for first in pairs:
        picked = [first]
        while len(picked) < n:
            end = parse(picked[-1])[1]
            nxt = [p for p in pairs if parse(p)[0] == end]
            if not nxt:
                break
            picked.append(nxt[0])
        if len(picked) == n:
            return picked
    raise SystemExit(f'no chain of {n}')


def place(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(META):
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def mirror_dir(src, dst):
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            place(p, os.path.join(dst, f))


os.makedirs(OUT, exist_ok=True)
lines = []
if KEY in 'KB':
    ifgs = f'{SRC}/interferograms'
    names = [d for d in os.listdir(ifgs) if re.fullmatch(r'\d+_\d+', d)]
    for name in chain(names, N):
        mirror_dir(f'{ifgs}/{name}', f'{OUT}/interferograms/{name}')
    mirror_dir(f'{SRC}/geometry', f'{OUT}/geometry')
if KEY == 'K':
    lines += [f'mintpy.load.unwFile = {OUT}/interferograms/*/filt_*.unw', f'mintpy.load.corFile = {OUT}/interferograms/*/filt_*.cor',
              f'mintpy.load.demFile = {OUT}/geometry/radar*.hgt', f'mintpy.load.lookupYFile = {OUT}/geometry/geomap*.trans',
              f'mintpy.load.lookupXFile = {OUT}/geometry/geomap*.trans']
if KEY == 'B':
    for f in os.listdir(SRC):
        if os.path.isfile(f'{SRC}/{f}'):
            shutil.copy2(f'{SRC}/{f}', f'{OUT}/{f}')
    lines += [f'mintpy.load.metaFile = {OUT}/supermaster.PRM', f'mintpy.load.baselineDir = {OUT}/baseline_table.dat',
              f'mintpy.load.unwFile = {OUT}/interferograms/*/unwrap_ll*.grd', f'mintpy.load.corFile = {OUT}/interferograms/*/corr_ll*.grd',
              f'mintpy.load.demFile = {OUT}/geometry/dem.grd', f'mintpy.load.incAngleFile = {OUT}/geometry/incidence_angle.grd',
              f'mintpy.load.azAngleFile = {OUT}/geometry/azimuth_angle.grd', f'mintpy.load.waterMaskFile = {OUT}/geometry/water_mask.grd']
if KEY == 'A':
    for d in ('unwrappedPhase', 'coherence', 'connectedComponents', 'DEM', 'incidenceAngle', 'azimuthAngle', 'lookAngle', 'mask'):
        if os.path.isdir(f'{SRC}/{d}') and not os.path.lexists(f'{OUT}/{d}'):
            os.symlink(f'{SRC}/{d}', f'{OUT}/{d}')
    os.makedirs(f'{OUT}/stack', exist_ok=True)
    roots = {n: ET.parse(f'{SRC}/stack/{n}.vrt').getroot() for n in ('unwrapStack', 'cohStack', 'connCompStack')}

    def band_pair(band):
        for md in band.findall('Metadata'):
            for mdi in md.findall('MDI'):
                if mdi.get('key') == 'Dates':
                    return mdi.text
        return None
    pairs = [band_pair(b) for b in roots['unwrapStack'].findall('VRTRasterBand')]
    def key(p):
        a, b = p.split('_')
        return tuple(sorted((a, b)))
    ordered = {f'{key(p)[0]}_{key(p)[1]}': p for p in pairs if p}
    picked = chain(list(ordered), N)
    keep = {key(ordered[p]) for p in picked}
    for n, root in roots.items():
        bands = root.findall('VRTRasterBand')
        kept = [b for b in bands if band_pair(b) and key(band_pair(b)) in keep]
        for b in bands:
            root.remove(b)
        for i, b in enumerate(kept, 1):
            b.set('band', str(i))
            root.append(b)
        ET.ElementTree(root).write(f'{OUT}/stack/{n}.vrt')
    dem = glob.glob(f'{OUT}/DEM/*.dem')[0]
    inc = glob.glob(f'{OUT}/incidenceAngle/*.vrt')[0]
    az = glob.glob(f'{OUT}/azimuthAngle/*.vrt')[0]
    wm = glob.glob(f'{OUT}/mask/*.msk')
    lines += ['mintpy.load.autoPath = no', f'mintpy.load.unwFile = {OUT}/stack/unwrapStack.vrt', f'mintpy.load.corFile = {OUT}/stack/cohStack.vrt',
              f'mintpy.load.connCompFile = {OUT}/stack/connCompStack.vrt', f'mintpy.load.demFile = {dem}',
              f'mintpy.load.incAngleFile = {inc}', f'mintpy.load.azAngleFile = {az}'] + ([f'mintpy.load.waterMaskFile = {wm[0]}'] if wm else [])
if KEY == 'F':
    ifgs = f'{SRC}/merged/interferograms'
    for name in chain(os.listdir(ifgs), N):
        mirror_dir(f'{ifgs}/{name}', f'{OUT}/merged/interferograms/{name}')
    mirror_dir(f'{SRC}/merged/geom_reference', f'{OUT}/merged/geom_reference')
    mirror_dir(f'{SRC}/reference', f'{OUT}/reference')
    os.symlink(f'{SRC}/baselines', f'{OUT}/baselines')
    lines += ['mintpy.load.autoPath = no', f'mintpy.load.metaFile = {OUT}/reference/IW*.xml', f'mintpy.load.baselineDir = {OUT}/baselines',
              f'mintpy.load.unwFile = {OUT}/merged/interferograms/*/filt_*.unw', f'mintpy.load.corFile = {OUT}/merged/interferograms/*/filt_*.cor',
              f'mintpy.load.connCompFile = {OUT}/merged/interferograms/*/filt_*.unw.conncomp',
              f'mintpy.load.demFile = {OUT}/merged/geom_reference/hgt.rdr', f'mintpy.load.lookupYFile = {OUT}/merged/geom_reference/lat.rdr',
              f'mintpy.load.lookupXFile = {OUT}/merged/geom_reference/lon.rdr', f'mintpy.load.incAngleFile = {OUT}/merged/geom_reference/los.rdr',
              f'mintpy.load.azAngleFile = {OUT}/merged/geom_reference/los.rdr', f'mintpy.load.shadowMaskFile = {OUT}/merged/geom_reference/shadowMask.rdr',
              f'mintpy.load.waterMaskFile = {OUT}/merged/geom_reference/waterMask.rdr']
open(f'{OUT}/load.txt', 'w').write('\n'.join(lines) + '\n')
print(KEY, N, 'ok', sorted(os.listdir(OUT)))

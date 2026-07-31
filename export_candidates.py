#!/usr/bin/env python3
"""
Export the NEW pond candidates (blobs > 0.7 probability that do not overlap an
already-mapped polygon) as ESRI Shapefiles.

Writes two versions:
  candidates/Pond_Candidates_2025_wgs84.shp        EPSG:4326 (lon/lat)
  candidates/Pond_Candidates_2025_magna.shp        MAGNA-SIRGAS 2018 / Origen
                                                   Nacional - same CRS as the
                                                   original Cuerpos_Agua file
plus a .zip of each for easy download from the web page.
"""
import json, os, shutil, sys
import shapefile                      # pyshp
from pyproj import Transformer

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'web', 'data', 'blobs_2025_high_70.json')
OUT   = os.path.join(HERE, 'web', 'candidates')
os.makedirs(OUT, exist_ok=True)

MAGNA = ("+proj=tmerc +lat_0=4 +lon_0=-73 +k=0.9992 +x_0=5000000 +y_0=2000000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
PRJ_WGS84 = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
             'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
PRJ_MAGNA = open(os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26.prj')).read()

if not os.path.exists(CACHE):
    sys.exit('No blob cache yet — click "Find blobs" in the viewer first.')

feats = [f for f in json.load(open(CACHE))['features'] if not f['properties']['known']]
print(f'new candidates to export: {len(feats)}')

to_magna = Transformer.from_crs('EPSG:4326', MAGNA, always_xy=True)


def write(path, prj_text, reproject):
    with shapefile.Writer(path, shapeType=shapefile.POLYGON) as w:
        w.field('cand_id',  'N', 8, 0)
        w.field('area_m2',  'N', 12, 0)
        w.field('area_ha',  'N', 12, 3)
        w.field('lon',      'N', 14, 6)
        w.field('lat',      'N', 14, 6)
        w.field('prob_min', 'N', 5, 2)      # blob threshold used
        w.field('year',     'N', 5, 0)
        w.field('source',   'C', 40)
        for i, f in enumerate(feats, 1):
            ring = f['geometry']['coordinates'][0]
            if reproject:
                ring = [to_magna.transform(x, y) for x, y in ring]
            # shapefile polygons are clockwise; GeoJSON rings here are CCW
            w.poly([[list(pt) for pt in reversed(ring)]])
            p = f['properties']
            w.record(i, int(p['area_m2']), round(p['area_m2'] / 10000, 3),
                     round(p['lon'], 6), round(p['lat'], 6), 0.70, 2025,
                     'RF on Satellite Embeddings, high penalty')
    with open(path + '.prj', 'w') as fh:
        fh.write(prj_text)
    base = os.path.basename(path)
    shutil.make_archive(path, 'zip', os.path.dirname(path),
                        base_dir=None, root_dir=None) if False else None
    return path


w1 = write(os.path.join(OUT, 'Pond_Candidates_2025_wgs84'), PRJ_WGS84, False)
w2 = write(os.path.join(OUT, 'Pond_Candidates_2025_magna'), PRJ_MAGNA, True)

# zip each shapefile set (shp+shx+dbf+prj) for one-click download
for base in (w1, w2):
    name = os.path.basename(base)
    tmp = os.path.join(OUT, '_pack')
    os.makedirs(tmp, exist_ok=True)
    for ext in ('.shp', '.shx', '.dbf', '.prj'):
        shutil.copy(base + ext, os.path.join(tmp, name + ext))
    shutil.make_archive(base, 'zip', tmp)
    shutil.rmtree(tmp)
    print('wrote', name + '.zip')

print('\nfiles in', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f, os.path.getsize(os.path.join(OUT, f)), 'bytes')

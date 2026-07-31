#!/usr/bin/env python3
"""
Export NEW pond candidates with their TRUE blob outlines (not bounding boxes).

Re-runs the vectorisation keeping the real polygon geometry, drops blobs that
overlap an already-mapped pond, and writes shapefiles in both WGS84 and
MAGNA-SIRGAS 2018 / Origen Nacional.
"""
import ee, json, os, shutil
import shapefile
from pyproj import Transformer

ee.Initialize(project='lofty-tokenizer-437115-e3')

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'web', 'candidates')
os.makedirs(OUT, exist_ok=True)
SEARCH = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56])
YEAR, THRESH, MIN_AREA, SCALE = 2025, 0.7, 2000, 20

_gj = json.load(open(os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26_wgs84.geojson')))
PONDS = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon(f['geometry']['coordinates']), {})
    for f in _gj['features']])

emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
       .filterDate(f'{YEAR}-01-01', f'{YEAR+1}-01-01').filterBounds(SEARCH).mosaic())
bands = emb.bandNames()
neg_pts = ee.FeatureCollection.randomPoints(region=SEARCH, points=25000, seed=101)  # high penalty
pos = (emb.sampleRegions(collection=PONDS, scale=10, tileScale=8)
       .randomColumn('r', 1).filter(ee.Filter.lt('r', 0.35)).map(lambda f: f.set('class', 1)))
neg = emb.sampleRegions(collection=neg_pts, scale=10, tileScale=8).map(lambda f: f.set('class', 0))
clf = (ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1)
       .setOutputMode('PROBABILITY').train(pos.merge(neg), 'class', bands))
prob = emb.classify(clf).rename('p').clip(SEARCH)

vec = (prob.gte(THRESH).selfMask().rename('b').reduceToVectors(
        geometry=SEARCH, scale=SCALE, geometryType='polygon', eightConnected=True,
        maxPixels=1e13, bestEffort=True, reducer=ee.Reducer.countEvery())
       .filter(ee.Filter.gte('count', max(2, MIN_AREA // (SCALE * SCALE))))
       .sort('count', False).limit(4000))

print('fetching outlines from Earth Engine…', flush=True)
raw = vec.getInfo()
print('blobs:', len(raw['features']))

# local overlap test against known ponds (bbox test, same as the viewer)
known = []
for f in _gj['features']:
    xs = [c[0] for c in f['geometry']['coordinates'][0]]
    ys = [c[1] for c in f['geometry']['coordinates'][0]]
    known.append((min(xs), min(ys), max(xs), max(ys)))

def overlaps(x0, y0, x1, y1):
    return any(not (a1 < x0 or a0 > x1 or b1 < y0 or b0 > y1) for a0, b0, a1, b1 in known)

cands = []
for f in raw['features']:
    g = f['geometry']
    rings = [g['coordinates'][0]] if g['type'] == 'Polygon' else \
            [part[0] for part in g['coordinates']]
    pts = [p for r in rings for p in r]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if overlaps(min(xs), min(ys), max(xs), max(ys)):
        continue
    cands.append({'rings': rings, 'area': f['properties']['count'] * SCALE * SCALE,
                  'lon': sum(xs) / len(xs), 'lat': sum(ys) / len(ys)})
cands.sort(key=lambda c: -c['area'])
print('new candidates:', len(cands))

MAGNA = ("+proj=tmerc +lat_0=4 +lon_0=-73 +k=0.9992 +x_0=5000000 +y_0=2000000 "
         "+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
PRJ_WGS84 = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
             'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
PRJ_MAGNA = open(os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26.prj')).read()
to_magna = Transformer.from_crs('EPSG:4326', MAGNA, always_xy=True)


def write(path, prj, reproject):
    with shapefile.Writer(path, shapeType=shapefile.POLYGON) as w:
        w.field('cand_id', 'N', 8, 0); w.field('area_m2', 'N', 12, 0)
        w.field('area_ha', 'N', 12, 3); w.field('lon', 'N', 14, 6)
        w.field('lat', 'N', 14, 6);     w.field('prob_min', 'N', 5, 2)
        w.field('year', 'N', 5, 0);     w.field('source', 'C', 40)
        for i, c in enumerate(cands, 1):
            rings = []
            for r in c['rings']:
                pts = [to_magna.transform(x, y) for x, y in r] if reproject else [tuple(p) for p in r]
                rings.append([list(p) for p in reversed(pts)])   # shapefile wants CW
            w.poly(rings)
            w.record(i, int(c['area']), round(c['area'] / 10000, 3),
                     round(c['lon'], 6), round(c['lat'], 6), THRESH, YEAR,
                     'RF Satellite Embeddings, high penalty')
    open(path + '.prj', 'w').write(prj)
    name = os.path.basename(path)
    tmp = os.path.join(OUT, '_pack'); os.makedirs(tmp, exist_ok=True)
    for ext in ('.shp', '.shx', '.dbf', '.prj'):
        shutil.copy(path + ext, os.path.join(tmp, name + ext))
    shutil.make_archive(path, 'zip', tmp); shutil.rmtree(tmp)
    print('wrote', name + '.zip')


write(os.path.join(OUT, 'Pond_Candidates_2025_outlines_wgs84'), PRJ_WGS84, False)
write(os.path.join(OUT, 'Pond_Candidates_2025_outlines_magna'), PRJ_MAGNA, True)
print('DONE')

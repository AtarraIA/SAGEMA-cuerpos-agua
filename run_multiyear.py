#!/usr/bin/env python3
"""
Multi-year pond-probability maps.

For every year the Satellite Embedding dataset covers, train a Random Forest on
the pond polygons against THAT year's embeddings, classify the whole search
region to a 0-1 probability, and save it as a grayscale PNG on an identical
pixel grid (pixel value / 255 = pond probability). Also saves a Sentinel-2
true-colour backdrop and a meta.json describing the grid.

These PNGs feed web/index.html (year switch / RGB composite / difference).
"""
import ee, json, os, urllib.request
from PIL import Image

ee.Initialize(project='lofty-tokenizer-437115-e3')

HERE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(HERE, 'web', 'data')
os.makedirs(DATA, exist_ok=True)

GEOJSON = os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26_wgs84.geojson')
SEARCH  = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56])
DIM     = 900                      # longest-side pixels; identical across years

# ---- labels ---------------------------------------------------------------
gj = json.load(open(GEOJSON))
feats = [ee.Feature(ee.Geometry.Polygon(f['geometry']['coordinates']),
                    {'Id': f['properties']['Id']}) for f in gj['features']]
ponds = ee.FeatureCollection(feats)

# ---- which years are available --------------------------------------------
col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(SEARCH)
years = sorted(col.aggregate_array('system:time_start')
               .map(lambda t: ee.Date(t).get('year')).distinct().getInfo())
print('years:', years)

# fixed negatives (same points every year -> comparable across time)
neg_pts = ee.FeatureCollection.randomPoints(region=SEARCH, points=9000, seed=101)

def pond_pixels(emb, sub, seed):
    return (emb.sampleRegions(collection=ponds, scale=10, tileScale=8)
            .randomColumn('r', seed).filter(ee.Filter.lt('r', sub))
            .map(lambda f: f.set('class', 1)))

def prob_image(year):
    emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .filterBounds(SEARCH).mosaic())
    bands = emb.bandNames()
    neg = emb.sampleRegions(collection=neg_pts, scale=10, tileScale=8) \
             .map(lambda f: f.set('class', 0))
    train = pond_pixels(emb, 0.35, 1).merge(neg)
    clf = (ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1)
           .setOutputMode('PROBABILITY').train(train, 'class', bands))
    return emb.classify(clf).rename('p').clip(SEARCH)

def save_png(image, viz, path):
    url = image.visualize(**viz).getThumbURL(
        {'region': SEARCH, 'dimensions': DIM, 'format': 'png'})
    urllib.request.urlretrieve(url, path)

# ---- per-year probability (grayscale = raw values) ------------------------
GRAY = {'min': 0, 'max': 1, 'palette': ['000000', 'ffffff']}
for y in years:
    out = os.path.join(DATA, f'prob_{y}.png')
    if os.path.exists(out):
        print('skip', y); continue
    print('processing', y, '...', flush=True)
    save_png(prob_image(y), GRAY, out)
    print('  wrote', out, flush=True)

# ---- Sentinel-2 true-colour backdrop (2024) -------------------------------
s2_path = os.path.join(DATA, 's2.png')
if not os.path.exists(s2_path):
    def maskS2(img):
        qa = img.select('QA60')
        m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate('2024-01-01', '2025-01-01').filterBounds(SEARCH)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2).median())
    save_png(s2, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, s2_path)
    print('wrote', s2_path)

# ---- meta.json (grid dimensions from an actual PNG) -----------------------
w, h = Image.open(os.path.join(DATA, f'prob_{years[0]}.png')).size
meta = {'years': years, 'width': w, 'height': h,
        'bbox': [-75.74, 9.14, -75.36, 9.56], 'hasS2': True}
json.dump(meta, open(os.path.join(DATA, 'meta.json'), 'w'), indent=2)
print('wrote meta.json:', meta)
print('DONE')

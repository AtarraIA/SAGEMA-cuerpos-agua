#!/usr/bin/env python3
"""
Local tile server for the pond viewer.

Serves the static page from web/ and exposes a small API that asks Earth Engine
for map-tile URLs. Tiles are rendered by Google at every zoom level, so the map
has full GEE quality instead of a fixed-resolution PNG.

  GET /api/init                      -> years, bbox, polygon geojson path
  GET /api/prob?year=2024            -> {url} pond-probability tiles
  GET /api/rgb?r=2017&g=2021&b=2025  -> {url} 3-year RGB composite
  GET /api/diff?a=2025&b=2017        -> {url} A-B difference
  GET /api/s2?year=2024              -> {url} Sentinel-2 true colour for that year

Run:  python3 serve.py     then open http://localhost:8777
"""
import ee, json, os, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from tile_cache import get_url

PROJECT = 'lofty-tokenizer-437115-e3'
HERE    = os.path.dirname(os.path.abspath(__file__))
WEB     = os.path.join(HERE, 'web')
GEOJSON = os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26_wgs84.geojson')
SEARCH  = None
PORT    = 8777

ee.Initialize(project=PROJECT)
SEARCH = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56])

# ---- labels ---------------------------------------------------------------
_gj = json.load(open(GEOJSON))
PONDS = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon(f['geometry']['coordinates']), {'Id': f['properties']['Id']})
    for f in _gj['features']])
NEG_PTS = ee.FeatureCollection.randomPoints(region=SEARCH, points=9000, seed=101)

MAGMA = ['000004', '3b0f70', '8c2981', 'de4968', 'fe9f6d', 'fcfdbf']
DIVERGE = ['2166ac', '67a9cf', 'd1e5f0', 'f7f7f7', 'fddbc7', 'ef8a62', 'b2182b']

_lock = threading.Lock()
_prob_cache = {}     # year -> ee.Image (probability)
_url_cache  = {}     # request key -> tile url


def embeddings(year):
    return (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate(f'{year}-01-01', f'{year+1}-01-01')
            .filterBounds(SEARCH).mosaic())


# False-positive penalty: Random Forest in EE has no class-weight parameter, so the
# cost-sensitive lever is the negative:positive ratio. More background examples raise
# the bar a pixel must clear to be called a pond, which suppresses false positives
# (measured on 2025's held-out split):
#   none   9,000 negatives -> FP 6.30%  precision 0.955  recall 0.874  acc 0.899
#   medium 16,000          -> FP 2.95%  precision 0.981  recall 0.832  acc 0.881
#   high   25,000          -> FP 1.75%  precision 0.988  recall 0.780  acc 0.852
PENALTY_NEG = {'none': 9000, 'medium': 16000, 'high': 25000}
PENALTY_YEARS = {2025}          # penalty is offered only for these years


def prob_image(year, penalty='none'):
    """Pond-probability image for a year (RF retrained on that year's embeddings)."""
    if year not in PENALTY_YEARS:
        penalty = 'none'
    key = (year, penalty)
    with _lock:
        if key in _prob_cache:
            return _prob_cache[key]
    emb = embeddings(year)
    bands = emb.bandNames()
    n_neg = PENALTY_NEG.get(penalty, 9000)
    neg_pts = (NEG_PTS if n_neg == 9000 else
               ee.FeatureCollection.randomPoints(region=SEARCH, points=n_neg, seed=101))
    pos = (emb.sampleRegions(collection=PONDS, scale=10, tileScale=8)
           .randomColumn('r', 1).filter(ee.Filter.lt('r', 0.35))
           .map(lambda f: f.set('class', 1)))
    neg = (emb.sampleRegions(collection=neg_pts, scale=10, tileScale=8)
           .map(lambda f: f.set('class', 0)))
    clf = (ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1)
           .setOutputMode('PROBABILITY').train(pos.merge(neg), 'class', bands))
    img = emb.classify(clf).rename('p').clip(SEARCH)
    with _lock:
        _prob_cache[key] = img
    return img


def s2_image(year):
    """Cloud-free Sentinel-2 composite. 10 m is the sensor's native limit, but a
    25th-percentile composite is markedly clearer than a median (less residual
    haze/cloud edge), which is the only real lever on apparent sharpness."""
    def mask(img):
        qa = img.select('QA60')
        clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        scl = img.select('SCL')
        good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))  # shadow/cloud/cirrus
        return img.updateMask(clear.And(good))
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .filterBounds(SEARCH)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
           .map(mask))
    return col.reduce(ee.Reducer.percentile([25])).rename(
        col.first().bandNames())


def tile_url(key, visual):
    """URL de teselas, con caché en memoria y en disco.

    La caché en disco (tile_cache.json) hace que reiniciar el servidor no
    vuelva a entrenar los modelos: las URLs ya calculadas se reutilizan.
    """
    with _lock:
        if key in _url_cache:
            return _url_cache[key]
    url = get_url(key, lambda: ee.data.getMapId({'image': visual})['tile_fetcher'].url_format)
    with _lock:
        _url_cache[key] = url
    return url


def blobs(year=2025, penalty='high', thresh=0.7, min_area=2000, limit=4000):
    """Bounding boxes around connected blobs of probability > thresh.

    Cached to disk (web/data/blobs_*.json) because vectorising the whole region
    is the slowest operation here.
    """
    cache = os.path.join(WEB, 'data', f'blobs_{year}_{penalty}_{int(thresh*100)}.json')
    if os.path.exists(cache):
        return json.load(open(cache))

    SCALE = 20            # vectorise at 20 m: fast, and plenty for bounding boxes
    p = prob_image(year, penalty)
    mask = p.gte(thresh).selfMask()

    # countEvery() gives a pixel count per blob for free -> area without any
    # per-feature geometry call (those are what make this slow)
    vec = mask.rename('b').reduceToVectors(
        geometry=SEARCH, scale=SCALE, geometryType='polygon',
        eightConnected=True, maxPixels=1e13, bestEffort=True,
        reducer=ee.Reducer.countEvery())
    # min_area also keeps the collection under EE's 5000-element query cap;
    # sorting by size first means the cap drops only the smallest specks
    min_px = max(2, int(min_area / (SCALE * SCALE)))
    vec = vec.filter(ee.Filter.gte('count', min_px)).sort('count', False).limit(limit)

    raw = vec.getInfo()          # single round-trip; everything below is local

    # known-pond bounding boxes for a cheap local overlap test
    known_boxes = []
    for f in _gj['features']:
        xs = [c[0] for c in f['geometry']['coordinates'][0]]
        ys = [c[1] for c in f['geometry']['coordinates'][0]]
        known_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    def overlaps_known(x0, y0, x1, y1):
        for a0, b0, a1, b1 in known_boxes:
            if not (a1 < x0 or a0 > x1 or b1 < y0 or b0 > y1):
                return 1
        return 0

    feats = []
    for f in raw['features']:
        coords = f['geometry']['coordinates']
        pts = coords[0] if f['geometry']['type'] == 'Polygon' else \
              [pt for part in coords for ring in part for pt in ring]
        xs = [c[0] for c in pts]; ys = [c[1] for c in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        npx = f['properties'].get('count', 0)
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            'properties': {
                'area_m2': npx * SCALE * SCALE,
                'known': overlaps_known(x0, y0, x1, y1),
                'lon': (x0 + x1) / 2, 'lat': (y0 + y1) / 2,
            }})

    feats.sort(key=lambda f: -f['properties']['area_m2'])
    for i, f in enumerate(feats):
        f['properties']['idx'] = i
    gj = {'type': 'FeatureCollection', 'features': feats}
    json.dump(gj, open(cache, 'w'))
    return gj


def available_years():
    col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(SEARCH)
    return sorted(col.aggregate_array('system:time_start')
                  .map(lambda t: ee.Date(t).get('year')).distinct().getInfo())


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith('/api/'):
            return super().do_GET()
        q = parse_qs(u.query)
        gi = lambda k, d=None: int(q[k][0]) if k in q else d
        try:
            if u.path == '/api/init':
                return self._json({
                    'years': available_years(),
                    'bbox': [-75.74, 9.14, -75.36, 9.56],
                    'penaltyYears': sorted(PENALTY_YEARS),
                    'penaltyStats': {   # measured on the held-out 20% split for 2025
                        'none':   {'neg': 9000,  'fp': 6.30, 'prec': 0.955, 'rec': 0.874, 'acc': 0.899},
                        'medium': {'neg': 16000, 'fp': 2.95, 'prec': 0.981, 'rec': 0.832, 'acc': 0.881},
                        'high':   {'neg': 25000, 'fp': 1.75, 'prec': 0.988, 'rec': 0.780, 'acc': 0.852},
                    }})

            if u.path == '/api/prob':
                y = gi('year')
                pen = q.get('pen', ['none'])[0]
                url = tile_url(f'prob{y}_{pen}', prob_image(y, pen).visualize(
                    min=0, max=1, palette=MAGMA))
                return self._json({'url': url, 'penalty': pen})

            if u.path == '/api/rgb':
                r, g, b = gi('r'), gi('g'), gi('b')
                img = ee.Image.cat(prob_image(r), prob_image(g), prob_image(b)) \
                        .rename(['r', 'g', 'b'])
                url = tile_url(f'rgb{r}_{g}_{b}',
                               img.visualize(bands=['r', 'g', 'b'], min=0, max=1))
                return self._json({'url': url})

            if u.path == '/api/diff':
                a, b = gi('a'), gi('b')
                d = prob_image(a).subtract(prob_image(b)).rename('d')
                url = tile_url(f'diff{a}_{b}',
                               d.visualize(min=-1, max=1, palette=DIVERGE))
                return self._json({'url': url})

            if u.path == '/api/blobs':
                y = gi('year', 2025)
                pen = q.get('pen', ['high'])[0]
                th = float(q.get('thresh', ['0.7'])[0])
                ma = float(q.get('minarea', ['2000'])[0])
                return self._json(blobs(y, pen, th, ma))

            if u.path == '/api/s2':
                y = gi('year')
                url = tile_url(f's2{y}', s2_image(y).visualize(
                    bands=['B4', 'B3', 'B2'], min=0, max=3000))
                return self._json({'url': url})

            return self._json({'error': 'unknown endpoint'}, 404)
        except Exception as e:
            return self._json({'error': str(e)}, 500)


if __name__ == '__main__':
    print(f'Pond viewer  ->  http://localhost:{PORT}')
    print('Earth Engine tiles are generated on demand; first load of a layer '
          'takes a few seconds while the model trains.')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()

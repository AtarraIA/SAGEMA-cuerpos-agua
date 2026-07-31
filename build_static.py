#!/usr/bin/env python3
"""
Construye el sitio estático (docs/) para GitHub Pages.

GitHub Pages sólo sirve archivos estáticos, así que no puede ejecutar serve.py.
Este script genera `docs/layers.json` con las URLs de teselas de Earth Engine
ya resueltas (son públicas) y copia todos los datos necesarios.

IMPORTANTE: las URLs de teselas de Earth Engine caducan (semanas). Para
refrescarlas basta con volver a ejecutar este script y hacer commit.
"""
import ee, json, os, shutil

ee.Initialize(project='lofty-tokenizer-437115-e3')

HERE = os.path.dirname(os.path.abspath(__file__))
WEB  = os.path.join(HERE, 'web')
DOCS = os.path.join(HERE, 'docs')
SEARCH = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56])

MAGMA = ['000004', '3b0f70', '8c2981', 'de4968', 'fe9f6d', 'fcfdbf']

_gj = json.load(open(os.path.join(HERE, 'Cuerpos_Agua_AtarraIA_29Julio26_wgs84.geojson')))
PONDS = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon(f['geometry']['coordinates']), {})
    for f in _gj['features']])

PENALTY_NEG = {'none': 9000, 'medium': 16000, 'high': 25000}


def embeddings(year):
    return (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate(f'{year}-01-01', f'{year+1}-01-01').filterBounds(SEARCH).mosaic())


def prob_image(year, penalty='none'):
    emb = embeddings(year); bands = emb.bandNames()
    n = PENALTY_NEG[penalty]
    pts = ee.FeatureCollection.randomPoints(region=SEARCH, points=n, seed=101)
    pos = (emb.sampleRegions(collection=PONDS, scale=10, tileScale=8)
           .randomColumn('r', 1).filter(ee.Filter.lt('r', 0.35))
           .map(lambda f: f.set('class', 1)))
    neg = emb.sampleRegions(collection=pts, scale=10, tileScale=8) \
             .map(lambda f: f.set('class', 0))
    clf = (ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1)
           .setOutputMode('PROBABILITY').train(pos.merge(neg), 'class', bands))
    return emb.classify(clf).rename('p').clip(SEARCH)


def s2_image(year):
    def mask(img):
        qa = img.select('QA60')
        clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        scl = img.select('SCL')
        good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(clear.And(good))
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(f'{year}-01-01', f'{year+1}-01-01').filterBounds(SEARCH)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).map(mask))
    return col.reduce(ee.Reducer.percentile([25])).rename(col.first().bandNames())


def url(image):
    return ee.data.getMapId({'image': image})['tile_fetcher'].url_format


def main():
    years = sorted(ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                   .filterBounds(SEARCH).aggregate_array('system:time_start')
                   .map(lambda t: ee.Date(t).get('year')).distinct().getInfo())
    print('años:', years)

    layers = {'years': years, 'bbox': [-75.74, 9.14, -75.36, 9.56],
              'prob': {}, 's2': {}, 'penaltyYears': [2025],
              'penaltyStats': {
                  'none':   {'neg': 9000,  'fp': 6.30, 'prec': 0.955, 'rec': 0.874, 'acc': 0.899},
                  'medium': {'neg': 16000, 'fp': 2.95, 'prec': 0.981, 'rec': 0.832, 'acc': 0.881},
                  'high':   {'neg': 25000, 'fp': 1.75, 'prec': 0.988, 'rec': 0.780, 'acc': 0.852}}}

    for y in years:
        print('  probabilidad', y, flush=True)
        layers['prob'][str(y)] = {'none': url(prob_image(y, 'none').visualize(
            min=0, max=1, palette=MAGMA))}
    for pen in ('medium', 'high'):
        print('  probabilidad 2025', pen, flush=True)
        layers['prob']['2025'][pen] = url(prob_image(2025, pen).visualize(
            min=0, max=1, palette=MAGMA))
    for y in years:
        print('  Sentinel-2', y, flush=True)
        layers['s2'][str(y)] = url(s2_image(y).visualize(
            bands=['B4', 'B3', 'B2'], min=0, max=3000))

    os.makedirs(DOCS, exist_ok=True)
    json.dump(layers, open(os.path.join(DOCS, 'layers.json'), 'w'), indent=1)
    print('escrito docs/layers.json')

    # copiar datos estáticos
    for sub in ('data', 'vendor', 'candidates'):
        s, d = os.path.join(WEB, sub), os.path.join(DOCS, sub)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
    # .nojekyll para que GitHub Pages no ignore carpetas con guion bajo
    open(os.path.join(DOCS, '.nojekyll'), 'w').close()
    print('datos copiados a docs/')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Measure the effect of hard-negative mining (false-positive penalty) for 2025."""
import ee, json
ee.Initialize(project='lofty-tokenizer-437115-e3')

SEARCH = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56])
gj = json.load(open('Cuerpos_Agua_AtarraIA_29Julio26_wgs84.geojson'))
PONDS = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon(f['geometry']['coordinates']), {'Id': f['properties']['Id']})
    for f in gj['features']])
YEAR = 2025

emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
       .filterDate(f'{YEAR}-01-01', f'{YEAR+1}-01-01').filterBounds(SEARCH).mosaic())
bands = emb.bandNames()

# 80/20 split by polygon
ps = PONDS.randomColumn('split', 42)
trainP = ps.filter(ee.Filter.lt('split', 0.8))
testP  = ps.filter(ee.Filter.gte('split', 0.8))

def pond_px(fc, sub, seed):
    return (emb.sampleRegions(collection=fc, scale=10, tileScale=8)
            .randomColumn('r', seed).filter(ee.Filter.lt('r', sub))
            .map(lambda f: f.set('class', 1)))

def land_px(n, seed):
    pts = ee.FeatureCollection.randomPoints(region=SEARCH, points=n, seed=seed)
    return emb.sampleRegions(collection=pts, scale=10, tileScale=8) \
              .map(lambda f: f.set('class', 0))

posTr, posTe = pond_px(trainP, 0.30, 1), pond_px(testP, 0.35, 2)
negTr, negTe = land_px(9000, 101), land_px(2000, 202)
testSet = posTe.merge(negTe)

def evaluate(name, clf):
    cm = testSet.classify(clf).errorMatrix('class', 'classification')
    acc = cm.accuracy().getInfo()
    m = cm.getInfo()
    tn, fp = m[0][0], m[0][1]
    fn, tp = m[1][0], m[1][1]
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn)
    print(f'{name:22} acc={acc:.4f}  FP-rate={fpr:.4f}  precision={prec:.4f}  recall={rec:.4f}')
    return acc, fpr, prec, rec

# ---------- baseline ----------
base = ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1) \
        .train(posTr.merge(negTr), 'class', bands)
evaluate('baseline', base)

# ---------- hard-negative mining ----------
# 1. score the landscape with a probability version of the baseline
baseProb = ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1) \
        .setOutputMode('PROBABILITY').train(posTr.merge(negTr), 'class', bands)
p0 = emb.classify(baseProb).rename('p')
pondMask = ee.Image().byte().paint(PONDS, 1).unmask(0).rename('pm')

# 2. find background pixels the model wrongly scores HIGH -> these are the errors
cand = ee.FeatureCollection.randomPoints(region=SEARCH, points=12000, seed=777)
scored = p0.addBands(pondMask).addBands(emb).sampleRegions(
    collection=cand, scale=10, tileScale=16)
hard = (scored.filter(ee.Filter.eq('pm', 0))
              .filter(ee.Filter.gt('p', 0.45))
              .map(lambda f: f.set('class', 0)))
print('hard negatives mined:', hard.size().getInfo())

# 3. retrain with hard negatives duplicated -> a harsher penalty on being wrong
for w in (2,):
    neg2 = negTr
    for _ in range(w):
        neg2 = neg2.merge(hard)
    clf = ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1) \
            .train(posTr.merge(neg2), 'class', bands)
    evaluate(f'hard-neg  weight x{w}', clf)

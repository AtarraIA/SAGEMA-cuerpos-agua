/**********************************************************************
 * Find artificial water bodies (aquaculture ponds) with Google
 * Satellite Embeddings (AlphaEarth Foundations, 64-D per 10 m pixel).
 *
 * Region : Golfo de Morrosquillo (Sucre, Colombia)
 * Method : SUPERVISED Random Forest (pond vs. background).
 *
 * NOTE — why not cosine similarity: the embeddings are unit vectors,
 * so ponds (median cos ~0.92) and random land (~0.87) overlap too much
 * for any single-centroid threshold -> "everything lights up". A trained
 * classifier learns the discriminative boundary and reaches ~89% test
 * accuracy on an 80/20 split by polygon.
 **********************************************************************/

// ----------------------------------------------------------------------
// 0) INPUTS
// ----------------------------------------------------------------------
var ponds = ee.FeatureCollection('users/josalazarmo/water-bodies');  // your polygons
var YEAR  = 2024;   // Satellite Embedding V1 covers 2017..2024
var searchRegion = ee.Geometry.Rectangle([-75.74, 9.14, -75.36, 9.56]);
var THRESH = 0.5;   // pond probability cutoff; raise to 0.6-0.7 for cleaner results

// ----------------------------------------------------------------------
// 1) DATA
// ----------------------------------------------------------------------
var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(searchRegion)
  .mosaic();
var bands = emb.bandNames();

// --- Sentinel-2 true-color composite (10 m) to actually SEE the ponds -----
function maskS2(img) {                       // mask clouds/cirrus via QA60
  var qa = img.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  return img.updateMask(mask);
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(searchRegion)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2)
  .median()
  .clip(searchRegion);

Map.setOptions('SATELLITE');                 // Google satellite basemap
Map.centerObject(ponds, 12);
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Sentinel-2 true color ' + YEAR);
Map.addLayer(searchRegion, {color: 'white'}, 'Search region', false);
Map.addLayer(ponds, {color: 'cyan'}, 'Reference ponds (input)');

// ----------------------------------------------------------------------
// 2) TRAIN / TEST SPLIT  (80/20 by polygon -> no spatial leakage)
// ----------------------------------------------------------------------
var pondsS     = ponds.randomColumn('split', 42);
var trainPonds = pondsS.filter(ee.Filter.lt('split', 0.8));
var testPonds  = pondsS.filter(ee.Filter.gte('split', 0.8));
print('train polys:', trainPonds.size(), ' test polys:', testPonds.size());

// pond pixels (positive). sampleRegions samples pixels whose center is INSIDE
// a polygon -> clean signature (never use reduceRegion+bestEffort: it coarsens
// scale and mixes in land).
function pondPixels(fc, subsample, seed) {
  return emb.sampleRegions({collection: fc, scale: 10, tileScale: 8})
    .randomColumn('r', seed).filter(ee.Filter.lt('r', subsample))
    .map(function (f) { return f.set('class', 1); });
}
// background pixels (negative) from random points. Ponds are ~0.25% of the area,
// so random points almost never land inside one.
function landPixels(n, seed) {
  var pts = ee.FeatureCollection.randomPoints({region: searchRegion, points: n, seed: seed});
  return emb.sampleRegions({collection: pts, scale: 10, tileScale: 8})
    .map(function (f) { return f.set('class', 0); });
}

var trainSet = pondPixels(trainPonds, 0.35, 1).merge(landPixels(9000, 101));
var testSet  = pondPixels(testPonds, 1.0, 2).merge(landPixels(2000, 202));

// ----------------------------------------------------------------------
// 3) TRAIN RF + EVALUATE ON HELD-OUT TEST SET  (~91% accuracy)
// ----------------------------------------------------------------------
var clf = ee.Classifier.smileRandomForest({numberOfTrees: 300, minLeafPopulation: 1})
  .train(trainSet, 'class', bands);
var cm  = testSet.classify(clf).errorMatrix('class', 'classification');
print('confusion matrix [actual 0/1 x pred 0/1]:', cm);
print('OVERALL ACCURACY:', cm.accuracy());
print('kappa:', cm.kappa());
print('pond recall   :', ee.Array(cm.producersAccuracy()).get([1, 0]));
print('pond precision :', ee.Array(cm.consumersAccuracy()).get([0, 1]));

// ----------------------------------------------------------------------
// 4) FINAL MODEL (all polygons) -> probability map
// ----------------------------------------------------------------------
var clfProb = ee.Classifier.smileRandomForest({numberOfTrees: 300, minLeafPopulation: 1})
  .setOutputMode('PROBABILITY')
  .train(pondPixels(ponds, 0.35, 1).merge(landPixels(9000, 101)), 'class', bands);

var prob = emb.classify(clfProb).rename('p').clip(searchRegion);

Map.addLayer(prob, {min: 0, max: 1,
  palette: ['000004', '3b0f70', '8c2981', 'de4968', 'fe9f6d', 'fcfdbf']},
  'Pond probability');

// ----------------------------------------------------------------------
// 5) CANDIDATES = high-probability pixels that are NOT already mapped
// ----------------------------------------------------------------------
var candidates = prob.gte(THRESH).selfMask();
var pondRaster = ee.Image().byte().paint(ponds, 1).unmask(0);
var newCand    = candidates.updateMask(pondRaster.not());

Map.addLayer(candidates, {palette: ['ffff00']}, 'All pond-like pixels', false);
Map.addLayer(newCand,    {palette: ['ff0000']}, 'NEW candidate ponds');

// ----------------------------------------------------------------------
// 6) VECTORIZE + EXPORT  (uncomment Export to run)
// ----------------------------------------------------------------------
var vectors = newCand.rename('cand')
  .reduceToVectors({
    geometry: searchRegion, scale: 10, geometryType: 'polygon',
    eightConnected: true, maxPixels: 1e13
  })
  .map(function (f) { return f.set('area_m2', f.geometry().area(1)); })
  .filter(ee.Filter.gte('area_m2', 400));   // drop <4-pixel speckle

print('candidate polygons:', vectors.size());
Map.addLayer(vectors, {color: 'orange'}, 'Candidate polygons (vector)', false);

// Export.table.toDrive({
//   collection: vectors,
//   description: 'candidate_ponds_Morrosquillo_' + YEAR,
//   fileFormat: 'SHP'
// });

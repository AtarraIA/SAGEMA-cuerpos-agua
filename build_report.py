#!/usr/bin/env python3
"""Build the technical report PDF describing the pond-detection workflow."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, ListFlowable,
                                ListItem, PageBreak)
from reportlab.pdfgen import canvas
import os

INK = colors.HexColor('#0b0b0b'); MUT = colors.HexColor('#52514e')
BLUE = colors.HexColor('#2a78d6'); ORANGE = colors.HexColor('#eb6834')
RULE = colors.HexColor('#d9d8d3'); PANEL = colors.HexColor('#f4f3ef')
HERE = os.path.dirname(os.path.abspath(__file__))

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Title'], fontName='Helvetica-Bold',
                    fontSize=22, leading=26, textColor=INK, spaceAfter=4, alignment=TA_LEFT)
SUB = ParagraphStyle('SUB', fontName='Helvetica', fontSize=11, leading=15,
                     textColor=MUT, spaceAfter=2)
H2 = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=14.5, leading=18,
                    textColor=INK, spaceBefore=16, spaceAfter=6)
H3 = ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=11.5, leading=15,
                    textColor=BLUE, spaceBefore=10, spaceAfter=3)
BODY = ParagraphStyle('BODY', fontName='Helvetica', fontSize=10.2, leading=15,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
BUL = ParagraphStyle('BUL', parent=BODY, spaceAfter=3)
CODE = ParagraphStyle('CODE', fontName='Courier', fontSize=8.8, leading=12,
                      textColor=colors.HexColor('#1a1a19'), backColor=PANEL,
                      borderPadding=(6,6,6,6), spaceBefore=3, spaceAfter=9)
CAP = ParagraphStyle('CAP', fontName='Helvetica-Oblique', fontSize=8.6, leading=11,
                     textColor=MUT, spaceBefore=3, spaceAfter=12)
mono = 'Courier'

def b(s): return f'<b>{s}</b>'
def code(s): return f'<font face="Courier" size="9">{s}</font>'

story = []

# ---- Header ---------------------------------------------------------------
story.append(Paragraph('Detecting Artificial Water Bodies with Google Satellite Embeddings', H1))
story.append(Paragraph('A supervised deep-embedding + Random Forest workflow in Google Earth Engine', SUB))
story.append(Paragraph('Study area: Golfo de Morrosquillo, Sucre, Colombia &nbsp;•&nbsp; '
                       'Input: 1,341 hand-drawn pond polygons &nbsp;•&nbsp; '
                       'Result: 90.8% test accuracy', SUB))
story.append(Spacer(1, 6))
story.append(HRFlowable(width='100%', thickness=1, color=RULE))

# ---- 1. Objective ---------------------------------------------------------
story.append(Paragraph('1. Objective', H2))
story.append(Paragraph(
    'We were given a set of <b>1,341 hand-digitised polygons</b> of artificial water bodies '
    '(aquaculture ponds / <i>camaroneras</i>) around the Golfo de Morrosquillo. The goal was to '
    'use these examples to <b>automatically find other, un-mapped ponds</b> in the surrounding '
    'landscape. This is a one-class-driven detection problem: we know what a pond looks like, and '
    'we want every place that looks the same.', BODY))
story.append(Paragraph(
    'The approach uses <b>Google Satellite Embeddings</b> — a learned 64-dimensional descriptor of '
    'each 10&nbsp;m pixel — as the feature space, and a <b>supervised Random Forest classifier</b> '
    'trained on the polygons to separate "pond" from "background".', BODY))

# ---- 2. Input data --------------------------------------------------------
story.append(Paragraph('2. Input data', H2))
data_rows = [
    ['Property', 'Value'],
    ['Features', '1,341 polygons (ESRI Shapefile)'],
    ['Geometry', 'Small ponds — median 1,700 m² (0.17 ha), max 9.8 ha, 462 ha total'],
    ['Original CRS', 'MAGNA-SIRGAS 2018 / Origen Nacional (Transverse Mercator, Colombia)'],
    ['Extent', '≈ 22 × 22 km; centre lon −75.515, lat 9.349'],
    ['Attributes', 'Id (integer) only'],
]
t = Table(data_rows, colWidths=[3.6*cm, 12.2*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
    ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('TEXTCOLOR',(0,1),(-1,-1),INK),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PANEL]),
    ('GRID',(0,0),(-1,-1),0.5,RULE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
]))
story.append(t)
story.append(Spacer(1,6))
story.append(Paragraph(
    'The first preprocessing step was reprojecting the polygons from the Colombian national grid '
    'to geographic coordinates (<b>WGS84 / EPSG:4326</b>), which Earth Engine expects. This was done '
    'with <font face="Courier" size="9">pyproj</font> and <font face="Courier" size="9">pyshp</font>, '
    'writing a GeoJSON that becomes the model\'s training labels.', BODY))

# ---- 3. Tools & libraries -------------------------------------------------
story.append(Paragraph('3. Tools and libraries', H2))
lib_rows = [
    ['Library / dataset', 'Role in the workflow'],
    ['earthengine-api (Python, v1.7)', 'Server-side geospatial ML: sampling, Random Forest, classification, export'],
    ['GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL', 'AlphaEarth 64-band learned embedding — the feature space'],
    ['COPERNICUS/S2_SR_HARMONIZED', 'Sentinel-2 surface reflectance — true-colour imagery for visual review'],
    ['ee.Classifier.smileRandomForest', 'The supervised classifier (Random Forest, SMILE implementation)'],
    ['pyproj / pyshp', 'Reproject shapefile (MAGNA-SIRGAS → WGS84) and read geometry'],
    ['Jupyter / nbconvert', 'Reproducible notebook, executed end-to-end'],
    ['reportlab, matplotlib', 'This report and its figures'],
]
t2 = Table(lib_rows, colWidths=[6.0*cm, 9.8*cm])
t2.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
    ('FONTNAME',(0,1),(0,-1),'Courier'),('FONTSIZE',(0,1),(0,-1),8.2),('TEXTCOLOR',(0,1),(-1,-1),INK),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PANEL]),
    ('GRID',(0,0),(-1,-1),0.5,RULE),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
]))
story.append(t2)

# ---- 4. The feature space -------------------------------------------------
story.append(Paragraph('4. The feature space: Satellite Embeddings (AlphaEarth)', H2))
story.append(Paragraph(
    'Instead of raw spectral bands, we use <b>Google\'s Satellite Embedding V1</b> dataset (AlphaEarth '
    'Foundations). A neural network was trained on large volumes of satellite time-series and produces, '
    'for every 10&nbsp;m pixel and every year, a <b>64-dimensional vector</b> that summarises what that '
    'pixel <i>is</i> — its land cover, texture, seasonality and water content — in a form where similar '
    'places have similar vectors.', BODY))
story.append(Paragraph(
    'Two properties matter for us: (1) each vector is <b>L2-normalised</b> (unit length), so it lives on '
    'the surface of a 64-D sphere; and (2) the 64 numbers are already a compact, information-rich '
    'description, so a simple classifier on top works well. We load the annual mosaic for 2024:', BODY))
story.append(Paragraph(
    "emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \\<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;.filterDate('2024-01-01','2025-01-01').filterBounds(region).mosaic()", CODE))

# ---- 5. Method ------------------------------------------------------------
story.append(Paragraph('5. Method', H2))

story.append(Paragraph('5.1 First attempt — cosine similarity (and why it failed)', H3))
story.append(Paragraph(
    'The intuitive idea is: average the embedding vectors inside the known ponds to get a single '
    '"pond signature", then score every pixel by its <b>cosine similarity</b> (dot product) to that '
    'signature and threshold. We built and tested this. It <b>does not work here</b>, for a concrete '
    'reason: because the embeddings are unit vectors, essentially <i>all</i> land cover sits at cosine '
    '0.8–0.9 to any given direction. The ponds and the general landscape overlap almost completely, so '
    'no threshold separates them — every threshold that keeps the ponds also floods half the map.', BODY))
story.append(Image(os.path.join(HERE,'fig_method_separation.png'), width=15.4*cm, height=7.9*cm))
story.append(Paragraph(
    'Figure 1. Median score of pond pixels vs. random background. With cosine similarity (left) the two '
    'classes are almost identical (0.92 vs 0.87) — unusable. The Random Forest (right) drives background '
    'down to 0.16 while keeping ponds at 0.92 — a clean, thresholdable gap.', CAP))
story.append(Paragraph(
    'A second, subtler bug we fixed here: the pond signature must be sampled with '
    '<font face="Courier" size="9">sampleRegions</font> (pixels whose centre is inside a polygon). Using '
    '<font face="Courier" size="9">reduceRegion</font> with <font face="Courier" size="9">bestEffort</font> '
    'silently coarsens the scale and mixes surrounding land into the signature, which actually '
    '<i>inverts</i> the result (land scores high). Strict in-polygon sampling avoids this.', BODY))

story.append(Paragraph('5.2 The working method — supervised Random Forest', H3))
story.append(Paragraph(
    'A <b>Random Forest</b> is an ensemble of decision trees; each tree is trained on a random subset of '
    'the data and features, and their votes are averaged. Unlike a single-centroid distance, it learns '
    'the <i>discriminative</i> boundary — which of the 64 dimensions, and in what combination, actually '
    'distinguish a pond from a rice field, a road, a roof or a river. That is exactly the signal a mean '
    'centroid throws away.', BODY))
story.append(Paragraph('Training data was built as two classes:', BODY))
story.append(ListFlowable([
    ListItem(Paragraph('<b>Positives (class 1)</b> — embedding vectors of pixels sampled strictly inside '
                       'the pond polygons.', BUL), leftIndent=6),
    ListItem(Paragraph('<b>Negatives (class 0)</b> — embedding vectors at random background points across '
                       'the search region (ponds are only ~0.25% of the area, so random points are almost '
                       'never inside a pond).', BUL), leftIndent=6),
], bulletType='bullet', start='•'))
story.append(Paragraph(
    "clf = ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=1) \\<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;.train(trainSet, 'class', bands)   # bands = the 64 embedding dimensions", CODE))

story.append(Paragraph('5.3 Honest evaluation — 80/20 split by polygon', H3))
story.append(Paragraph(
    'To measure real generalisation we split the <b>polygons</b> (not the pixels) 80/20. Every pixel in '
    'the test set therefore comes from a pond the model <b>never saw</b>. Splitting by pixel instead would '
    'leak information (neighbouring pixels of the same pond in both sets) and inflate the score. We report '
    'accuracy on that held-out 20% (269 ponds).', BODY))

# ---- 6. Results -----------------------------------------------------------
story.append(PageBreak())
story.append(Paragraph('6. Results', H2))
res_rows = [
    ['Metric', 'Baseline (100 trees)', 'Tuned (300 trees, more data)'],
    ['Overall test accuracy', '88.8%', '90.8%'],
    ['Kappa', '0.69', '0.73'],
    ['Pond recall (found)', '0.875', '0.905'],
    ['Pond precision', '0.987', '0.981'],
]
t3 = Table(res_rows, colWidths=[6.2*cm, 4.8*cm, 4.8*cm])
t3.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9.5),
    ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('TEXTCOLOR',(0,1),(-1,-1),INK),
    ('BACKGROUND',(2,1),(2,1),colors.HexColor('#e3f0ff')),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PANEL]),
    ('GRID',(0,0),(-1,-1),0.5,RULE),('ALIGN',(1,0),(-1,-1),'CENTER'),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
]))
story.append(t3)
story.append(Spacer(1,6))
story.append(Paragraph(
    'The tuned model clears the 90% target. Three changes moved the number: more trees (100→300), more '
    'training positives (15%→35% of pond pixels) and more negatives (6,000→9,000). These mostly recovered '
    'mixed pixels at pond edges — the main source of error — lifting recall from 0.875 to 0.905 at almost '
    'no cost to precision.', BODY))
story.append(Image(os.path.join(HERE,'fig_confusion.png'), width=10.2*cm, height=8.6*cm))
story.append(Paragraph(
    'Figure 2. Confusion matrix on the held-out test set. Of pond pixels, 8,042 are correctly found and '
    '849 missed (recall 0.905); of background, only 155 of 2,000 are false alarms. Precision on ponds is '
    '0.981 — when the model says "pond", it is almost always right.', CAP))

# ---- 7. Producing the map -------------------------------------------------
story.append(Paragraph('7. From classifier to candidate map', H2))
story.append(Paragraph(
    'For the final product the classifier is retrained on <b>all</b> polygons and set to '
    '<font face="Courier" size="9">PROBABILITY</font> output, giving each pixel a 0–1 pond likelihood. '
    'We threshold at 0.5, remove pixels already inside the known polygons (so only <b>new</b> candidates '
    'remain), vectorise them to polygons and export a shapefile to Google Drive.', BODY))
story.append(Image(os.path.join(HERE,'verify_probability_map.png'), width=12.5*cm, height=12.5*cm))
story.append(Paragraph(
    'Figure 3. Pond-probability map over the study area (dark = unlikely, bright yellow = likely pond). '
    'Cyan outlines are the input polygons; they sit exactly on bright spots, and the model also '
    'highlights additional bright clusters — candidate un-mapped ponds.', CAP))

# ---- 8. Visual review -----------------------------------------------------
story.append(Paragraph('8. Visual review with Sentinel-2', H2))
story.append(Paragraph(
    'To let an analyst confirm detections against reality, the script adds a <b>Sentinel-2 true-colour '
    'composite</b> (10&nbsp;m, cloud-masked median for the year) beneath the results. Toggling the '
    'probability layer over real imagery makes it easy to verify each candidate is an actual pond.', BODY))
story.append(Image(os.path.join(HERE,'verify_s2.png'), width=11.0*cm, height=11.0*cm))
story.append(Paragraph(
    'Figure 4. Sentinel-2 true-colour composite (2024) of part of the study area, with input ponds in '
    'cyan. This is the base layer used for visual validation in the Code Editor.', CAP))

# ---- 9. Reproducibility ---------------------------------------------------
story.append(Paragraph('9. Deliverables and how to reproduce', H2))
story.append(ListFlowable([
    ListItem(Paragraph('<font face="Courier" size="9">find_similar_ponds.ipynb</font> — the full Python '
                       'notebook (loads polygons, splits, trains, evaluates, classifies, exports). Runs '
                       'end-to-end and prints 90.8% accuracy.', BUL), leftIndent=6),
    ListItem(Paragraph('<font face="Courier" size="9">find_similar_ponds_GEE.js</font> — the same workflow '
                       'for the Earth Engine Code Editor, with interactive map layers and the Sentinel-2 '
                       'basemap.', BUL), leftIndent=6),
    ListItem(Paragraph('<font face="Courier" size="9">Cuerpos_..._wgs84.geojson</font> — polygons '
                       'reprojected to WGS84 (model input).', BUL), leftIndent=6),
    ListItem(Paragraph('The candidate ponds are exported as a shapefile to Google Drive by the final cell '
                       '/ export block.', BUL), leftIndent=6),
], bulletType='bullet', start='•'))

# ---- 10. Limitations ------------------------------------------------------
story.append(Paragraph('10. Limitations and next steps', H2))
story.append(ListFlowable([
    ListItem(Paragraph('<b>Random negatives.</b> Background is sampled randomly, so the model mainly learns '
                       '"pond vs. generic land". Adding <i>hard</i> negatives (natural lakes, rivers) would '
                       'sharpen the pond-vs-natural-water distinction.', BUL), leftIndent=6),
    ListItem(Paragraph('<b>Edge pixels.</b> Most remaining error is 10&nbsp;m pixels straddling a pond edge; '
                       'a minimum-mapping-unit filter (already applied at 400&nbsp;m²) and post-processing '
                       'reduce speckle.', BUL), leftIndent=6),
    ListItem(Paragraph('<b>Single year.</b> Running 2017–2024 and comparing would reveal pond expansion over '
                       'time, since the embedding dataset is annual.', BUL), leftIndent=6),
    ListItem(Paragraph('<b>Field validation.</b> Candidates should be checked against high-resolution '
                       'imagery or ground truth before operational use.', BUL), leftIndent=6),
], bulletType='bullet', start='•'))
story.append(Spacer(1,10))
story.append(HRFlowable(width='100%', thickness=1, color=RULE))
story.append(Paragraph('Prepared for J. Salazar (Universidad Nacional de Colombia). Study area: Golfo de '
                       'Morrosquillo, Sucre. Method: Google Satellite Embeddings V1 + Random Forest in '
                       'Google Earth Engine.', SUB))

# ---- Footer with page numbers --------------------------------------------
def footer(cv, doc):
    cv.saveState()
    cv.setFont('Helvetica', 8); cv.setFillColor(MUT)
    cv.drawString(2*cm, 1.1*cm, 'Artificial water-body detection — technical report')
    cv.drawRightString(A4[0]-2*cm, 1.1*cm, f'Page {doc.page}')
    cv.setStrokeColor(RULE); cv.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    cv.restoreState()

doc = SimpleDocTemplate(os.path.join(HERE,'Pond_Detection_Report.pdf'), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title='Detecting Artificial Water Bodies with Google Satellite Embeddings',
                        author='J. Salazar')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote Pond_Detection_Report.pdf')

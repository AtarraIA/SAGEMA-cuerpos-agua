# Detección de cuerpos de agua artificiales — Golfo de Morrosquillo

Detección automática de estanques acuícolas (camaroneras) en el Golfo de
Morrosquillo (Sucre, Colombia) a partir de **1.341 polígonos dibujados a mano**,
usando **Google Satellite Embeddings** (AlphaEarth) y un clasificador
**Random Forest** en Google Earth Engine.

**Visor en línea:** https://USUARIO.github.io/REPOSITORIO/

---

## Resultados

| Métrica | Valor |
|---|---|
| Exactitud sobre el 20 % de prueba | **90,8 %** |
| Kappa | 0,73 |
| Exhaustividad (recall) de estanques | 0,905 |
| Precisión de estanques | 0,981 |
| Años procesados | 2017 – 2025 (9 años) |
| Candidatos nuevos detectados (2025) | **3.130** |

La división 80/20 se hace **por polígono**, no por píxel, de modo que cada píxel
de prueba proviene de un estanque que el modelo nunca vio.

### Por qué Random Forest y no similitud coseno

Los *embeddings* son vectores unitarios, así que **toda** la cobertura terrestre
queda en un coseno de 0,8–0,9 respecto a cualquier dirección: los estanques
(mediana 0,92) y el terreno (mediana 0,87) se solapan casi por completo y ningún
umbral los separa. Un clasificador supervisado sí aprende la frontera
discriminante y alcanza ~91 % de exactitud.

---

## El visor

Tres modos:

- **Un año** — mapa de probabilidad (escala magma) por año, con teselas de
  Earth Engine a resolución completa.
- **RGB (3 años)** — cada año en un canal de color. Gris/blanco = estanque
  estable en los tres años; un color = presente sobre todo en ese año.
- **Diferencia** — A − B en escala divergente. Azul = ganado, rojo = perdido.

Además:

- **Mapa base** submétrico (Google / Esri) o **Sentinel-2 por año** (10 m) para
  ver el terreno como estaba en cada fecha.
- **Penalización de falsos positivos** (sólo 2025): al aumentar la proporción de
  muestras de fondo se eleva el listón para declarar un estanque.

  | Nivel | Muestras de fondo | Falsos positivos | Precisión | Exhaustividad |
  |---|---|---|---|---|
  | Ninguna | 9.000 | 6,30 % | 0,955 | 0,874 |
  | Media | 16.000 | 2,95 % | 0,981 | 0,832 |
  | Alta | 25.000 | 1,75 % | 0,988 | 0,780 |

- **Detecciones**: recuadros rojos sobre manchas con P > 0,7, con navegación
  «Siguiente / Anterior» (teclas `N` y `P`) para revisarlas una por una.
- **Descarga de candidatos** en shapefile (contornos reales y recuadros), en
  WGS84 y en MAGNA-SIRGAS 2018 / Origen Nacional.

---

## Estructura

```
docs/            sitio estático publicado en GitHub Pages
  index.html     el visor (en español)
  layers.json    URLs de teselas de Earth Engine ya resueltas
  data/          PNG por año, polígonos, manchas detectadas
  candidates/    shapefiles de candidatos nuevos (.zip)
serve.py         servidor local con backend de Earth Engine (versión completa)
build_static.py  genera docs/layers.json  ← ejecutar para refrescar teselas
run_multiyear.py genera los PNG de probabilidad por año
export_outlines.py / export_candidates.py   exportación a shapefile
find_similar_ponds.ipynb   cuaderno reproducible del método
Pond_Detection_Report.pdf  informe técnico detallado
```

---

## Uso local (versión completa, con backend)

```bash
pip install earthengine-api pyshp pyproj
earthengine authenticate
python3 serve.py           # http://localhost:8777
```

El backend genera las capas bajo demanda, así que permite cualquier combinación
de años y umbrales.

## Refrescar el sitio estático

> **Las URLs de teselas de Earth Engine caducan** (semanas). Cuando el visor
> publicado deje de mostrar las capas, regenera `layers.json`:

```bash
earthengine authenticate     # si hace falta
python3 build_static.py
git add docs/layers.json && git commit -m "Refrescar teselas" && git push
```

---

## Método, en resumen

1. Reproyectar los polígonos de MAGNA-SIRGAS 2018 a WGS84.
2. Cargar `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64 bandas, 10 m) del año.
3. Muestrear píxeles **dentro** de los polígonos con `sampleRegions`
   (positivos) y puntos aleatorios de fondo (negativos).
4. Entrenar `ee.Classifier.smileRandomForest(300 árboles)` en modo probabilidad.
5. Clasificar la región, umbralizar, vectorizar y exportar.

> Nota metodológica: usar `reduceRegion` con `bestEffort` para construir la
> firma degrada la escala y mezcla terreno circundante, lo que **invierte** el
> resultado. Hay que muestrear con `sampleRegions`.

---

## Créditos

- Polígonos de referencia: proyecto PAE.
- Datos: Google Satellite Embeddings (AlphaEarth), Copernicus Sentinel-2.
- Procesamiento: Google Earth Engine.

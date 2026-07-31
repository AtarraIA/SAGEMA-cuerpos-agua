#!/usr/bin/env python3
"""
Caché persistente de URLs de teselas de Earth Engine.

Entrenar el Random Forest y registrar la capa en Earth Engine es lo caro del
proceso. Las URLs resultantes son públicas y siguen siendo válidas durante
semanas, así que se guardan en disco: al reiniciar `serve.py` o volver a
ejecutar `build_static.py` se reutilizan en lugar de recalcularse.

Uso:
    from tile_cache import get_url
    url = get_url('prob_2025_high', lambda: ee.data.getMapId(...)...)

Para forzar el recálculo: borrar tile_cache.json, o llamar con force=True,
o `python3 tile_cache.py --check` para validar y purgar las caducadas.
"""
import json, os, time, urllib.request

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'tile_cache.json')
MAX_AGE_DAYS = 21          # las URLs de EE suelen durar semanas


def _load():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))
        except Exception:
            return {}
    return {}


def _save(d):
    tmp = CACHE + '.tmp'
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, CACHE)


def get_url(key, make_url, force=False, max_age_days=MAX_AGE_DAYS):
    """Devuelve la URL de teselas para `key`, calculándola sólo si hace falta.

    `make_url` es una función sin argumentos que devuelve la URL (la parte cara).
    """
    d = _load()
    e = d.get(key)
    if e and not force:
        age = (time.time() - e.get('created', 0)) / 86400
        if age < max_age_days:
            return e['url']
    url = make_url()
    d[key] = {'url': url, 'created': time.time()}
    _save(d)
    return url


def tile_ok(url_template, z=13, x=2400, y=3800, timeout=25):
    """Comprueba que una URL de teselas sigue respondiendo."""
    u = url_template.replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))
    try:
        with urllib.request.urlopen(u, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def prune(check=False):
    """Elimina entradas caducadas (por edad, o por no responder si check=True)."""
    d = _load(); out = {}; dropped = []
    for k, e in d.items():
        age = (time.time() - e.get('created', 0)) / 86400
        ok = age < MAX_AGE_DAYS and (not check or tile_ok(e['url']))
        if ok:
            out[k] = e
        else:
            dropped.append(k)
    _save(out)
    return dropped


if __name__ == '__main__':
    import sys
    d = _load()
    print(f'{len(d)} URLs en caché ({CACHE})')
    for k, e in sorted(d.items()):
        age = (time.time() - e.get('created', 0)) / 86400
        print(f'  {k:28} {age:5.1f} días')
    if '--check' in sys.argv:
        bad = prune(check=True)
        print('purgadas:', bad if bad else 'ninguna')

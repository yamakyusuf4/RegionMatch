import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from math import isfinite

repo=Path(r'c:\Users\ayush\RegionMatch')
feat=joblib.load(repo / 'models' / 'model_features.joblib')
path=None
candidates=[repo / 'data' / 'processed' / 'training_data_geo.csv', repo / 'training_data_geo.csv', repo / 'data' / 'processed' / 'training_data_clean.csv', repo / 'training_data_v1.csv']
for p in candidates:
    if p.exists():
        path=p
        break
if path is None:
    raise SystemExit('Dataset not found')
DF=pd.read_csv(path)
pipe=joblib.load(repo / 'models' / 'location_model.joblib')
X=DF.reindex(columns=feat).copy()
for c in X.columns:
    X[c]=pd.to_numeric(X[c], errors='coerce')
X=X.fillna(X.median(numeric_only=True)).fillna(0)
base=pipe.predict(X)

# helper haversine
import numpy as _np

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = _np.radians(_np.asarray(lat1, dtype=float))
    lon1 = _np.radians(_np.asarray(lon1, dtype=float))
    lat2 = _np.radians(float(lat2))
    lon2 = _np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = _np.sin(dlat/2)**2 + _np.cos(lat1)*_np.cos(lat2)*_np.sin(dlon/2)**2
    return 2 * R * _np.arcsin(_np.sqrt(a))

# cities list (subset used by app)
UK_CITIES = [
    ("London", -0.1276, 51.5072),
    ("Birmingham", -1.8904, 52.4862),
    ("Manchester", -2.2426, 53.4808),
    ("Leeds", -1.5491, 53.8008),
    ("Liverpool", -2.9916, 53.4084),
]

print('City, n_candidates_within_50km, base_std_within, base_min, base_max, industry_boost_std (core_tech)')
for (city, lng, lat) in UK_CITIES:
    if 'lad_lat' not in DF.columns or 'lad_lng' not in DF.columns:
        print(city, 'NO CENTROIDS')
        continue
    dist = haversine_km(DF['lad_lat'].fillna(lat).values, DF['lad_lng'].fillna(lng).values, lat, lng)
    mask = dist <= 50.0
    if mask.sum() == 0:
        print(city, 0)
        continue
    base_sub = base[mask]
    core_tech = pd.to_numeric(DF.loc[mask, 'core_tech_density'], errors='coerce').fillna(0).values
    print(city, int(mask.sum()), round(float(base_sub.std()),6), round(float(base_sub.min()),6), round(float(base_sub.max()),6), round(float(_np.std(core_tech)),6))

print('\nSummary global base std:', float(base.std()))

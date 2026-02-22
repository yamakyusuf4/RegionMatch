import joblib
import pandas as pd
from pathlib import Path

repo=Path(r'c:\Users\ayush\RegionMatch')
feat=joblib.load(repo / 'models' / 'model_features.joblib')
print('Loaded feature list length:', len(feat))

path=None
candidates=[repo / 'data' / 'processed' / 'training_data_geo.csv', repo / 'training_data_geo.csv', repo / 'data' / 'processed' / 'training_data_clean.csv', repo / 'training_data_v1.csv']
for p in candidates:
    if p.exists():
        path=p
        break
print('Using dataset:', path)

if path is None:
    raise SystemExit('Dataset not found')

DF=pd.read_csv(path)
print('Columns count:', len(DF.columns))

industry_col_map = {
    'Technology': 'core_tech_density',
    'Creative': 'creative_density',
    'Innovation': 'innovation_density',
    'Business Services': 'business_services_density',
    'Retail/Hospitality': 'business_density',
    'Industrial/Logistics': 'business_density'
}

for k,v in industry_col_map.items():
    if v in DF.columns:
        s=pd.to_numeric(DF[v], errors='coerce')
        print(f"{v}: present, min={s.min():.6f}, max={s.max():.6f}, nunique={s.nunique(dropna=True)}")
    else:
        print(f"{v}: MISSING")

pipe=joblib.load(repo / 'models' / 'location_model.joblib')
X=DF.reindex(columns=feat).copy()
for c in X.columns:
    X[c]=pd.to_numeric(X[c], errors='coerce')
X=X.fillna(X.median(numeric_only=True)).fillna(0)
base=pipe.predict(X)
print('base min/max/std:', float(base.min()), float(base.max()), float(base.std()))

# Show how urgency/industry would affect top-5 differences for two industries and urgencies
import numpy as np
from math import isfinite

def compute_scores(industry, urgency):
    industry_col = industry_col_map.get(industry)
    if industry_col and industry_col in DF.columns:
        industry_boost = (pd.to_numeric(DF[industry_col], errors='coerce') - pd.to_numeric(DF[industry_col], errors='coerce').min()) / (pd.to_numeric(DF[industry_col], errors='coerce').max() - pd.to_numeric(DF[industry_col], errors='coerce').min() + 1e-9)
        industry_boost = industry_boost.fillna(0).values
    else:
        industry_boost = np.zeros(len(DF))
    urgency_factor_map = {'<3 months': 1.0, '3-6 months': 0.6, '6+ months': 0.3}
    urgency_factor = urgency_factor_map.get(urgency, 0.6)
    adj = urgency_factor * 0.20 * industry_boost
    base_std = float(np.std(base)) + 1e-9
    adj_std = float(np.std(adj)) + 1e-9
    adj_scaled = adj * (base_std / adj_std) if adj_std>1e-12 else np.zeros_like(adj)
    final_score = 0.50 * base + 0.50 * adj_scaled
    final_norm = 100 * (final_score - final_score.min()) / (final_score.max() - final_score.min() + 1e-9)
    return final_norm

s1 = compute_scores('Technology', '<3 months')
s2 = compute_scores('Creative', '<3 months')
print('Top 5 (Technology, <3m):', list(DF['lad_name'][np.argsort(-s1)][:5].values))
print('Top 5 (Creative, <3m):', list(DF['lad_name'][np.argsort(-s2)][:5].values))

# Compare if top5 lists are identical
print('Top5 identical?', list(DF['lad_name'][np.argsort(-s1)][:5].values) == list(DF['lad_name'][np.argsort(-s2)][:5].values))

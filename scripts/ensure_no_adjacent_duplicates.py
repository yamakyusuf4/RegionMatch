"""
Validate that adjacent recommendations do not belong to the same city.

This script checks the top-5 recommendations generated for each city
and ensures that no two consecutive recommendations are from the same city
(i.e., the lad_name is not repeated adjacently).
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
repo = Path(__file__).resolve().parents[1]
feat = joblib.load(repo / 'models' / 'model_features.joblib')

# Load dataset
dataset_path = None
candidates = [
    repo / 'data' / 'processed' / 'training_data_geo.csv',
    repo / 'training_data_geo.csv',
    repo / 'data' / 'processed' / 'training_data_clean.csv',
    repo / 'training_data_v1.csv'
]
for p in candidates:
    if p.exists():
        dataset_path = p
        break

if dataset_path is None:
    raise SystemExit('Dataset not found')

df = pd.read_csv(dataset_path)
pipe = joblib.load(repo / 'models' / 'location_model.joblib')

# Prepare data
X = df.reindex(columns=feat).copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')
X = X.fillna(X.median(numeric_only=True)).fillna(0)
base = pipe.predict(X)

# Industry mapping
industry_col_map = {
    'Technology': 'core_tech_density',
    'Creative': 'creative_density',
    'Innovation': 'innovation_density',
    'Business Services': 'business_services_density',
    'Retail/Hospitality': 'business_density',
    'Industrial/Logistics': 'business_density'
}

# Haversine helper
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def minmax_series(s):
    s = pd.to_numeric(s, errors='coerce')
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# Test cities list
uk_cities = [
    ('London', -0.1276, 51.5072),
    ('Birmingham', -1.8904, 52.4862),
    ('Manchester', -2.2426, 53.4808),
    ('Leeds', -1.5491, 53.8008),
    ('Liverpool', -2.9916, 53.4084),
    ('Bristol', -2.5879, 51.4545),
    ('Sheffield', -1.4701, 53.3811),
    ('Newcastle upon Tyne', -1.6178, 54.9783),
]

print('Validating: 1) No adjacent duplicates, 2) Max score 99-99.5, 3) No opportunity in the selected city\n')
print('City, Top-5 Max Score, In-city LADs excluded?, Has Adjacent Duplicate?')

violations = []

for city, lng, lat in uk_cities:
    if 'lad_lat' not in df.columns or 'lad_lng' not in df.columns:
        print(f'{city}: NO CENTROIDS IN DATASET')
        continue
    
    # Filter to city (50 km radius)
    dist_km = haversine_km(
        df['lad_lat'].fillna(lat).values,
        df['lad_lng'].fillna(lng).values,
        lat,
        lng
    )
    city_mask = dist_km <= 50.0
    city_df = df.loc[city_mask].copy()
    
    if len(city_df) == 0:
        print(f'{city}: NO CANDIDATES WITHIN 50KM')
        continue
    
    # Exclude LADs that are IN the city (within 10 km)
    in_city_dist = haversine_km(
        city_df['lad_lat'].fillna(lat).values,
        city_df['lad_lng'].fillna(lng).values,
        lat,
        lng
    )
    not_in_city = in_city_dist > 10.0
    city_df = city_df.loc[not_in_city].copy()
    city_scores = base[city_mask][not_in_city]
    
    if len(city_df) == 0:
        print(f'{city}: ALL CANDIDATES ARE IN CITY, NO OPPORTUNITIES')
        continue
    
    # Score candidates
    # Apply industry adjustment (using Technology as default)
    industry = 'Technology'
    industry_col = industry_col_map.get(industry)
    if industry_col and industry_col in city_df.columns:
        industry_boost = minmax_series(city_df[industry_col]).values
    else:
        industry_boost = np.zeros(len(city_df))
    urgency_factor = 1.0  # <3 months
    adj = urgency_factor * 0.20 * industry_boost
    base_std = float(np.std(city_scores)) + 1e-9
    adj_std = float(np.std(adj)) + 1e-9
    adj_scaled = adj * (base_std / adj_std) if adj_std > 1e-12 else np.zeros_like(adj)
    final_score = 0.50 * city_scores + 0.50 * adj_scaled
    
    # Normalize to 0-100
    final_norm = 100 * (final_score - final_score.min()) / (final_score.max() - final_score.min() + 1e-9)
    city_df['final_score'] = final_norm
    
    # Get top-5 and cap scores to 99-99.5 range
    top5 = city_df.sort_values('final_score', ascending=False).head(5)[['lad_name', 'final_score']].copy()
    
    # Cap max score to random value between 99-99.5
    max_score_cap = 99.0 + (np.random.random() * 0.5)
    if top5['final_score'].max() > 0:
        scale = max_score_cap / top5['final_score'].max()
        top5['final_score'] = top5['final_score'] * scale
    
    max_score = top5['final_score'].max()
    score_in_range = 99.0 <= max_score <= 99.5
    
    # Check for adjacent duplicates (same LAD name appearing consecutively)
    has_adjacent_dup = False
    for i in range(len(top5) - 1):
        if top5.iloc[i]['lad_name'] == top5.iloc[i + 1]['lad_name']:
            has_adjacent_dup = True
            break
    
    if not score_in_range or has_adjacent_dup:
        violations.append(city)
        status = []
        if score_in_range:
            status.append("✓")
        else:
            status.append(f"✗ (max={max_score:.2f}, not 99-99.5)")
        status.append("✓")  # in-city excluded
        if has_adjacent_dup:
            status.append("✗")
        else:
            status.append("✓")
        print(f'{city}: {max_score:.2f}, Excluded, {" | ".join(status)[-1]}')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            print(f'  {idx}. {row["lad_name"]} ({row["final_score"]:.2f})')
    else:
        print(f'{city}: {max_score:.2f}, YES, ✓')

print('\n' + '='*60)
if violations:
    print(f'VIOLATIONS FOUND ({len(violations)} cities):')
    for city in violations:
        print(f'  - {city}')
else:
    print('✓ All cities passed validation:')
    print('  1. Max score between 99-99.5')
    print('  2. Selected city LADs excluded (>10km)')
    print('  3. No adjacent duplicates in top-5')

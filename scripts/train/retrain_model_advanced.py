"""
Enhanced Location Model Retraining Script

This script trains multiple regression models on training_data_geo.csv:
- Linear Regression (OLS multiple regression)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1+L2)
- Random Forest (ensemble method)
- Gradient Boosting (ensemble method)

Compares performance and saves the best model.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETUP
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "processed" / "training_data_geo.csv"
MODEL_SAVE_PATH = REPO_ROOT / "models" / "location_model.joblib"
FEATURES_SAVE_PATH = REPO_ROOT / "models" / "model_features.joblib"

print("=" * 80)
print("ADVANCED LOCATION MODEL RETRAINING - MULTIPLE ALGORITHMS")
print("=" * 80)

# ============================================================
# LOAD DATA
# ============================================================
print(f"\n[STEP 1] Loading training data from {DATA_PATH}...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# FEATURE SELECTION
# ============================================================
print(f"\n[STEP 2] Selecting features...")

exclude_cols = {
    'lad_code', 'lad_name', 'council_id', 'lad_lat', 'lad_lng', 'target_score'
}

all_cols = set(df.columns)
feature_cols = sorted(list(all_cols - exclude_cols))

print(f"✓ Selected {len(feature_cols)} features")
print(f"  Features: {', '.join(feature_cols[:5])}... (and {len(feature_cols)-5} more)")

# ============================================================
# PREPARE DATA
# ============================================================
print(f"\n[STEP 3] Preparing training data...")

X = df[feature_cols].copy()
y = df['target_score'].copy()

# Remove missing targets
valid_mask = ~y.isnull()
X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)} samples")

# ============================================================
# BUILD MODELS
# ============================================================
print(f"\n[STEP 4] Training multiple models...")

models = {
    'Linear Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    'Lasso Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.001, random_state=42, max_iter=5000))
    ]),
    'Elastic Net': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.005, l1_ratio=0.5, random_state=42, max_iter=5000))
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
    ]),
    'Gradient Boosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42))
    ])
}

results = {}

for name, pipeline in models.items():
    print(f"\n  Training {name}...")
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    # Test predictions
    y_pred_test = pipeline.predict(X_test)
    
    # Metrics
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'pipeline': pipeline,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
    
    print(f"    ✓ CV R² (mean): {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    print(f"    ✓ Test R²: {test_r2:.6f}")
    print(f"    ✓ Test RMSE: {test_rmse:.6f}")
    print(f"    ✓ Test MAE: {test_mae:.6f}")

# ============================================================
# COMPARE & SELECT BEST
# ============================================================
print(f"\n[STEP 5] Comparing model performance...")
print("\nModel Rankings (by Test R²):")
print("-" * 80)

sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)

for rank, (name, metrics) in enumerate(sorted_models, 1):
    print(f"{rank}. {name:25s} | R²: {metrics['test_r2']:.6f} | RMSE: {metrics['test_rmse']:.6f}")

best_name, best_metrics = sorted_models[0]
print("-" * 80)
print(f"\n✓ BEST MODEL: {best_name}")
print(f"  Test R² Score: {best_metrics['test_r2']:.6f}")
print(f"  Test RMSE: {best_metrics['test_rmse']:.6f}")
print(f"  CV R² (mean): {best_metrics['cv_mean']:.6f}")

# ============================================================
# SAVE BEST MODEL
# ============================================================
print(f"\n[STEP 6] Saving best model...")

best_pipeline = best_metrics['pipeline']
joblib.dump(best_pipeline, MODEL_SAVE_PATH)
joblib.dump(feature_cols, FEATURES_SAVE_PATH)

print(f"✓ Model saved to {MODEL_SAVE_PATH}")
print(f"✓ Features saved to {FEATURES_SAVE_PATH}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("RETRAINING COMPLETE")
print("=" * 80)
print(f"\nFinal Model Configuration:")
print(f"  • Algorithm: {best_name}")
print(f"  • Features: {len(feature_cols)}")
print(f"  • Training samples: {len(X)}")
print(f"  • Test R² Score: {best_metrics['test_r2']:.6f}")
print(f"  • Test RMSE: {best_metrics['test_rmse']:.6f}")
print(f"  • CV Mean R²: {best_metrics['cv_mean']:.6f}")
print(f"\nModels tested: {len(models)}")
for i, (name, metrics) in enumerate(sorted_models, 1):
    print(f"  {i}. {name:25s} (R²: {metrics['test_r2']:.6f})")
print(f"\nTo use the new model, restart the Streamlit app:")
print(f"  python -m streamlit run app/app.py")
print("=" * 80)

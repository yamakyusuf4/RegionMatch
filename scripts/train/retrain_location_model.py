"""
Retrain the location model using training_data_geo.csv

This script:
1. Loads the training data with geo features
2. Selects appropriate features (excludes identifiers, locations, and target)
3. Trains a scikit-learn Pipeline with Ridge regression
4. Evaluates with cross-validation and test split
5. Saves the retrained model and feature list to models/
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETUP
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from scripts/train/
DATA_PATH = REPO_ROOT / "data" / "processed" / "training_data_geo.csv"
MODEL_SAVE_PATH = REPO_ROOT / "models" / "location_model.joblib"
FEATURES_SAVE_PATH = REPO_ROOT / "models" / "model_features.joblib"

print("=" * 70)
print("LOCATION MODEL RETRAINING SCRIPT")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================
print(f"\n[1/5] Loading data from {DATA_PATH}...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# FEATURE SELECTION
# ============================================================
print(f"\n[2/5] Selecting features...")

# Columns to exclude (identifiers, locations, and target)
exclude_cols = {
    'lad_code',                  # LAD identifier
    'lad_name',                  # LAD name (identifier)
    'council_id',                # Council identifier
    'lad_lat',                   # Location
    'lad_lng',                   # Location
    'target_score',              # Target variable
}

# Identify feature columns
all_cols = set(df.columns)
feature_cols = sorted(list(all_cols - exclude_cols))

print(f"✓ Selected {len(feature_cols)} features (excluded {len(exclude_cols)} non-feature columns)")
print(f"\nFeatures:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# ============================================================
# PREPARE DATA
# ============================================================
print(f"\n[3/5] Preparing data for training...")

X = df[feature_cols].copy()
y = df['target_score'].copy()

# Check for missing values
missing_count = X.isnull().sum().sum()
print(f"✓ Feature matrix shape: {X.shape}")
print(f"  Missing values in features: {missing_count}")
print(f"  Missing values in target: {y.isnull().sum()}")

# Remove rows with missing target
valid_mask = ~y.isnull()
X = X.loc[valid_mask].copy()
y = y.loc[valid_mask].copy()
print(f"✓ After removing missing targets: {len(X)} samples")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# ============================================================
# BUILD & TRAIN PIPELINE
# ============================================================
print(f"\n[4/5] Building and training pipeline...")

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0, random_state=42))
])

# Train on full data for final model
print(f"  Training on full dataset ({len(X)} samples)...")
pipeline.fit(X, y)
print(f"✓ Pipeline trained")

# ============================================================
# EVALUATION
# ============================================================
print(f"\n[5/5] Evaluating model...")

# Cross-validation on full data (5-fold)
cv_scores = cross_val_score(
    pipeline, X, y, 
    cv=5, 
    scoring='r2',
    n_jobs=-1
)
print(f"  Cross-validation (5-fold) R² scores: {cv_scores}")
print(f"  Mean CV R²: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")

# Evaluate on test set
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n  Train Set Metrics:")
print(f"    R² Score:  {train_r2:.6f}")
print(f"    RMSE:      {train_rmse:.6f}")
print(f"    MAE:       {train_mae:.6f}")

print(f"\n  Test Set Metrics:")
print(f"    R² Score:  {test_r2:.6f}")
print(f"    RMSE:      {test_rmse:.6f}")
print(f"    MAE:       {test_mae:.6f}")

# ============================================================
# SAVE MODEL & FEATURES
# ============================================================
print(f"\n[SAVING] Storing model and features...")

# Save pipeline
joblib.dump(pipeline, MODEL_SAVE_PATH)
print(f"✓ Model saved to {MODEL_SAVE_PATH}")

# Save feature list
joblib.dump(feature_cols, FEATURES_SAVE_PATH)
print(f"✓ Features saved to {FEATURES_SAVE_PATH}")

print("\n" + "=" * 70)
print("RETRAINING COMPLETE")
print("=" * 70)
print(f"\nSummary:")
print(f"  • Model: Ridge Regression (alpha=1.0)")
print(f"  • Features: {len(feature_cols)}")
print(f"  • Training samples: {len(X)}")
print(f"  • Test R² Score: {test_r2:.6f}")
print(f"  • CV Mean R²: {cv_scores.mean():.6f}")
print(f"\nNext: Restart the Streamlit app to use the new model:")
print(f"  python -m streamlit run app/app.py")
print("=" * 70)

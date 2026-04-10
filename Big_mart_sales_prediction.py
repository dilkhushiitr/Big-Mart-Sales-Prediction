
# BigMart Sales Prediction 


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS    = [42, 123, 999]   # multi-seed for variance reduction
N_SPLITS = 5

# ═══════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════
train = pd.read_csv('/Users/Dilkhush1/Downloads/ABB/train_v9rqX0R.csv')
test  = pd.read_csv('/Users/Dilkhush1/Downloads/ABB/test_AbJTz2l.csv')

n_train = len(train)
test['Item_Outlet_Sales'] = np.nan
data = pd.concat([train, test], ignore_index=True)

# ═══════════════════════════════════════════════════════
# 2. MISSING VALUES
# ═══════════════════════════════════════════════════════
item_avg_weight = data.groupby('Item_Identifier')['Item_Weight'].mean()
data['Item_Weight'] = data['Item_Weight'].fillna(
    data['Item_Identifier'].map(item_avg_weight))

outlet_size_mode = (
    data.dropna(subset=['Outlet_Size'])
    .groupby('Outlet_Type')['Outlet_Size']
    .agg(lambda x: x.mode()[0])
)
mask = data['Outlet_Size'].isna()
data.loc[mask, 'Outlet_Size'] = data.loc[mask, 'Outlet_Type'].map(outlet_size_mode)

data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan)
item_avg_vis = data.groupby('Item_Identifier')['Item_Visibility'].mean()
data['Item_Visibility'] = data['Item_Visibility'].fillna(
    data['Item_Identifier'].map(item_avg_vis))

# ═══════════════════════════════════════════════════════
# 3. BASE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════
fat_map = {'low fat':'Low Fat','LF':'Low Fat','reg':'Regular',
           'Regular':'Regular','Low Fat':'Low Fat'}
data['Item_Fat_Content'] = data['Item_Fat_Content'].map(fat_map)
data['Item_Category']    = data['Item_Identifier'].str[:2]
data.loc[data['Item_Category'] == 'NC', 'Item_Fat_Content'] = 'Non-Edible'

data['Outlet_Age']      = 2013 - data['Outlet_Establishment_Year']
data['Is_New_Outlet']   = (data['Outlet_Age'] < 10).astype(int)
data['Is_Grocery']      = (data['Outlet_Type'] == 'Grocery Store').astype(int)

# Visibility
data['Item_Visibility_MeanRatio'] = (data['Item_Visibility'] /
                                      data['Item_Identifier'].map(item_avg_vis))
data['Outlet_Mean_Visibility'] = (data.groupby('Outlet_Identifier')['Item_Visibility']
                                     .transform('mean'))
data['Visibility_vs_Outlet']   = data['Item_Visibility'] / data['Outlet_Mean_Visibility']

# MRP
data['Item_MRP_log']           = np.log1p(data['Item_MRP'])
data['Price_Per_Weight']       = data['Item_MRP'] / data['Item_Weight']
data['MRP_Outlet_Percentile']  = (data.groupby('Outlet_Identifier')['Item_MRP']
                                      .rank(pct=True))
data['MRP_Category_Percentile']= (data.groupby('Item_Category')['Item_MRP']
                                      .rank(pct=True))
data['MRP_Cluster'] = pd.cut(
    data['Item_MRP'], bins=[0,50,100,150,200,250,300],
    labels=[0,1,2,3,4,5]).astype(float)

# Outlet aggregates
data['Outlet_Item_Count']    = (data.groupby('Outlet_Identifier')['Item_Identifier']
                                    .transform('count'))
data['Outlet_Mean_MRP']      = (data.groupby('Outlet_Identifier')['Item_MRP']
                                    .transform('mean'))
data['Outlet_MRP_Std']       = (data.groupby('Outlet_Identifier')['Item_MRP']
                                    .transform('std'))
data['Category_Mean_MRP']    = (data.groupby('Item_Category')['Item_MRP']
                                    .transform('mean'))
data['Item_MRP_vs_Category'] = data['Item_MRP'] / data['Category_Mean_MRP']
data['Item_Outlet_Spread']   = (data.groupby('Item_Identifier')['Outlet_Identifier']
                                    .transform('nunique'))

for col in ['Item_Type','Outlet_Identifier','Item_Category','Item_Identifier']:
    data[f'{col}_count'] = data.groupby(col)[col].transform('count')

# ── Outlet sales stats from TRAIN only ─────────────────
train_df = data.iloc[:n_train].copy()
test_df  = data.iloc[n_train:].copy()

outlet_sales_stats = train_df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg(
    Outlet_Sales_Mean='mean', Outlet_Sales_Std='std', Outlet_Sales_Median='median'
).reset_index()

item_sales_stats = train_df.groupby('Item_Identifier')['Item_Outlet_Sales'].agg(
    Item_Sales_Mean='mean', Item_Sales_Std='std'
).reset_index()

for df in [train_df, test_df]:
    df['Outlet_Sales_Mean']   = df['Outlet_Identifier'].map(
        outlet_sales_stats.set_index('Outlet_Identifier')['Outlet_Sales_Mean'])
    df['Outlet_Sales_Std']    = df['Outlet_Identifier'].map(
        outlet_sales_stats.set_index('Outlet_Identifier')['Outlet_Sales_Std'])
    df['Outlet_Sales_Median'] = df['Outlet_Identifier'].map(
        outlet_sales_stats.set_index('Outlet_Identifier')['Outlet_Sales_Median'])
    df['Item_Sales_Mean']     = df['Item_Identifier'].map(
        item_sales_stats.set_index('Item_Identifier')['Item_Sales_Mean'])
    df['Item_Sales_Std']      = df['Item_Identifier'].map(
        item_sales_stats.set_index('Item_Identifier')['Item_Sales_Std'])

# Fill test NaN sales stats with global mean
global_mean_sales = train_df['Item_Outlet_Sales'].mean()
for col in ['Outlet_Sales_Mean','Outlet_Sales_Std','Outlet_Sales_Median',
            'Item_Sales_Mean','Item_Sales_Std']:
    test_df[col] = test_df[col].fillna(global_mean_sales)

data = pd.concat([train_df, test_df], ignore_index=True)

# ═══════════════════════════════════════════════════════
# 4. SMOOTH TARGET ENCODING   
# ═══════════════════════════════════════════════════════
kf_te  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
SMOOTH = 20

train_df = data.iloc[:n_train].copy()
test_df  = data.iloc[n_train:].copy()
global_mean = train_df['Item_Outlet_Sales'].mean()

def smooth_te(key_series, target_series, global_mean, smooth=20):
    stats = target_series.groupby(key_series).agg(['mean','count'])
    smoothed = ((stats['count'] * stats['mean'] + smooth * global_mean)
                / (stats['count'] + smooth))
    return smoothed

# Single-column TEs
single_te_cols = ['Item_Type','Outlet_Identifier','Item_Category',
                  'Outlet_Type','Outlet_Location_Type','Item_Identifier']

for col in single_te_cols:
    train_df[f'{col}_TE'] = 0.0
    for tr_idx, val_idx in kf_te.split(train_df):
        tr_s, val_s = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        sm = smooth_te(tr_s[col], tr_s['Item_Outlet_Sales'], global_mean, SMOOTH)
        train_df.iloc[val_idx, train_df.columns.get_loc(f'{col}_TE')] = (
            val_s[col].map(sm).fillna(global_mean).values)
    sm_full = smooth_te(train_df[col], train_df['Item_Outlet_Sales'], global_mean, SMOOTH)
    test_df[f'{col}_TE'] = test_df[col].map(sm_full).fillna(global_mean)

# ── Cross TEs (most powerful feature!) ─────────────────
# Item × Outlet  (direct history lookup — huge signal)
cross_pairs = [
    ('Item_Identifier',  'Outlet_Identifier'),   # ← #1 feature typically
    ('Item_Identifier',  'Outlet_Type'),
    ('Item_Category',    'Outlet_Identifier'),
    ('Item_Category',    'Outlet_Type'),
    ('Item_Type',        'Outlet_Type'),
    ('Item_Fat_Content', 'Outlet_Type'),
]

for col_a, col_b in cross_pairs:
    cross_key = f'{col_a}_x_{col_b}'
    train_df[cross_key] = train_df[col_a].astype(str) + '_' + train_df[col_b].astype(str)
    test_df[cross_key]  = test_df[col_a].astype(str)  + '_' + test_df[col_b].astype(str)

    te_name = f'{cross_key}_TE'
    train_df[te_name] = 0.0
    for tr_idx, val_idx in kf_te.split(train_df):
        tr_s, val_s = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        sm = smooth_te(tr_s[cross_key], tr_s['Item_Outlet_Sales'], global_mean, SMOOTH)
        train_df.iloc[val_idx, train_df.columns.get_loc(te_name)] = (
            val_s[cross_key].map(sm).fillna(global_mean).values)
    sm_full = smooth_te(train_df[cross_key], train_df['Item_Outlet_Sales'], global_mean, SMOOTH)
    test_df[te_name] = test_df[cross_key].map(sm_full).fillna(global_mean)

    # Drop the string key column — only keep the TE
    train_df.drop(columns=[cross_key], inplace=True)
    test_df.drop(columns=[cross_key], inplace=True)

# ═══════════════════════════════════════════════════════
# 5. LABEL ENCODING
# ═══════════════════════════════════════════════════════
data = pd.concat([train_df, test_df], ignore_index=True)

categorical_cols = ['Item_Fat_Content','Item_Type','Outlet_Identifier',
                    'Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Category']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# Interaction features
data['MRP_x_OutletType']   = data['Item_MRP']   * data['Outlet_Type']
data['MRP_x_Visibility']   = data['Item_MRP']   * data['Item_Visibility']
data['Age_x_Visibility']   = data['Outlet_Age'] * data['Item_Visibility']
data['MRP_x_OutletAge']    = data['Item_MRP']   * data['Outlet_Age']
data['Weight_x_Visibility']= data['Item_Weight']* data['Item_Visibility']
data['Fat_x_MRP']          = data['Item_Fat_Content'] * data['Item_MRP']
data['OutletType_x_Age']   = data['Outlet_Type'] * data['Outlet_Age']
data['MRP_x_OutletMeanSales'] = data['Item_MRP'] * data['Outlet_Sales_Mean']

# ═══════════════════════════════════════════════════════
# 6. PREPARE TRAIN / TEST
# ═══════════════════════════════════════════════════════
drop_cols    = ['Item_Identifier','Outlet_Establishment_Year','Item_Outlet_Sales']
feature_cols = [c for c in data.columns if c not in drop_cols]

train_df = data.iloc[:n_train]
test_df  = data.iloc[n_train:]

X      = train_df[feature_cols].reset_index(drop=True)
y      = np.log1p(train_df['Item_Outlet_Sales'].values)
X_test = test_df[feature_cols].reset_index(drop=True)

print(f"Features: {X.shape[1]}  |  Train: {X.shape[0]}  |  Test: {X_test.shape[0]}\n")

# ═══════════════════════════════════════════════════════
# 7. OPTUNA: TUNE LIGHTGBM
# ═══════════════════════════════════════════════════════
print("=== Optuna tuning LightGBM (50 trials) ===")

def lgb_objective(trial):
    params = dict(
        n_estimators      = 3000,
        learning_rate     = trial.suggest_float('lr', 0.003, 0.05, log=True),
        num_leaves        = trial.suggest_int('num_leaves', 31, 255),
        max_depth         = trial.suggest_int('max_depth', 4, 10),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 60),
        feature_fraction  = trial.suggest_float('feature_fraction', 0.5, 1.0),
        bagging_fraction  = trial.suggest_float('bagging_fraction', 0.5, 1.0),
        bagging_freq      = 1,
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = -1,
    )
    kf_opt  = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_rmse = []
    for tr_idx, val_idx in kf_opt.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        oof_rmse.append(np.sqrt(mean_squared_error(y[val_idx], m.predict(X.iloc[val_idx]))))
    return np.mean(oof_rmse)

study = optuna.create_study(direction='minimize')
study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
best_lgb = study.best_params
print(f"Best LGB params: {best_lgb}")
print(f"Best LGB CV RMSE: {study.best_value:.5f}\n")

# ═══════════════════════════════════════════════════════
# 8. MULTI-SEED CROSS-VALIDATED BASE MODELS
# ═══════════════════════════════════════════════════════

def rmse(a, b): return np.sqrt(mean_squared_error(a, b))

# ── Helper: run one model with multiple seeds ──────────
def multi_seed_cv(make_model_fn, seeds, X, y, X_test, n_splits=N_SPLITS, label=''):
    all_oof  = []
    all_pred = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof  = np.zeros(len(X))
        pred = np.zeros(len(X_test))
        for tr_idx, val_idx in kf.split(X):
            m = make_model_fn(seed)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            oof[val_idx] = m.predict(X.iloc[val_idx])
            pred += m.predict(X_test) / n_splits
        fold_rmse = rmse(y, oof)
        print(f"  {label} seed={seed}  OOF RMSE(log): {fold_rmse:.5f}")
        all_oof.append(oof)
        all_pred.append(pred)
    avg_oof  = np.mean(all_oof, axis=0)
    avg_pred = np.mean(all_pred, axis=0)
    cv = rmse(y, avg_oof)
    print(f"  {label} multi-seed OOF RMSE: {cv:.5f}\n")
    return avg_oof, avg_pred, cv

# ── LightGBM with early stopping (special handling) ───
print("=== LightGBM (Optuna-tuned, multi-seed) ===")
lgb_oof_all, lgb_pred_all = [], []
for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof  = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X):
        m = lgb.LGBMRegressor(
            n_estimators      = 5000,
            learning_rate     = best_lgb['lr'],
            num_leaves        = best_lgb['num_leaves'],
            max_depth         = best_lgb['max_depth'],
            min_child_samples = best_lgb['min_child_samples'],
            feature_fraction  = best_lgb['feature_fraction'],
            bagging_fraction  = best_lgb['bagging_fraction'],
            bagging_freq      = 1,
            reg_alpha         = best_lgb['reg_alpha'],
            reg_lambda        = best_lgb['reg_lambda'],
            random_state      = seed,
            n_jobs            = -1,
            verbosity         = -1,
        )
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[val_idx] = m.predict(X.iloc[val_idx])
        pred += m.predict(X_test) / N_SPLITS
    print(f"  LGB seed={seed}  OOF: {rmse(y, oof):.5f}")
    lgb_oof_all.append(oof)
    lgb_pred_all.append(pred)

lgb_oof  = np.mean(lgb_oof_all,  axis=0)
lgb_pred = np.mean(lgb_pred_all, axis=0)
lgb_cv   = rmse(y, lgb_oof)
print(f"  LGB multi-seed OOF: {lgb_cv:.5f}\n")

# ── XGBoost (multi-seed with early stopping) ──────────
print("=== XGBoost (multi-seed) ===")
xgb_oof_all, xgb_pred_all = [], []
for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof  = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X):
        m = xgb.XGBRegressor(
            n_estimators        = 5000,
            learning_rate       = 0.007,
            max_depth           = 5,
            min_child_weight    = 5,
            subsample           = 0.75,
            colsample_bytree    = 0.7,
            colsample_bylevel   = 0.7,
            reg_alpha           = 0.1,
            reg_lambda          = 1.0,
            gamma               = 0.05,
            early_stopping_rounds = 150,
            random_state        = seed,
            verbosity           = 0,
            n_jobs              = -1,
        )
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              verbose=False)
        oof[val_idx] = m.predict(X.iloc[val_idx])
        pred += m.predict(X_test) / N_SPLITS
    print(f"  XGB seed={seed}  OOF: {rmse(y, oof):.5f}")
    xgb_oof_all.append(oof)
    xgb_pred_all.append(pred)

xgb_oof  = np.mean(xgb_oof_all,  axis=0)
xgb_pred = np.mean(xgb_pred_all, axis=0)
xgb_cv   = rmse(y, xgb_oof)
print(f"  XGB multi-seed OOF: {xgb_cv:.5f}\n")

# ── CatBoost (multi-seed) ─────────────────────────────
print("=== CatBoost (multi-seed) ===")

def make_cat(seed):
    return CatBoostRegressor(
        iterations          = 5000,
        learning_rate       = 0.02,
        depth               = 6,
        l2_leaf_reg         = 3,
        min_data_in_leaf    = 5,
        bagging_temperature = 0.5,
        random_strength     = 0.5,
        border_count        = 128,
        loss_function       = 'RMSE',
        early_stopping_rounds = 150,
        random_seed         = seed,
        verbose             = 0,
    )

cat_oof_all, cat_pred_all = [], []
for seed in SEEDS:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof  = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X):
        m = make_cat(seed)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=(X.iloc[val_idx], y[val_idx]),
              use_best_model=True)
        oof[val_idx] = m.predict(X.iloc[val_idx])
        pred += m.predict(X_test) / N_SPLITS
    print(f"  CAT seed={seed}  OOF: {rmse(y, oof):.5f}")
    cat_oof_all.append(oof)
    cat_pred_all.append(pred)

cat_oof  = np.mean(cat_oof_all,  axis=0)
cat_pred = np.mean(cat_pred_all, axis=0)
cat_cv   = rmse(y, cat_oof)
print(f"  CAT multi-seed OOF: {cat_cv:.5f}\n")

# ── ExtraTrees (multi-seed) ───────────────────────────
print("=== ExtraTrees (multi-seed) ===")

def make_et(seed):
    return ExtraTreesRegressor(
        n_estimators    = 1000,
        max_features    = 0.6,
        min_samples_leaf= 5,
        n_jobs          = -1,
        random_state    = seed,
    )

et_oof, et_pred, et_cv = multi_seed_cv(make_et, SEEDS, X, y, X_test, label='ET')

# ── MLP (scaled, multi-seed) ──────────────────────────
print("=== MLP (multi-seed) ===")

imputer  = SimpleImputer(strategy="median")
scaler   = StandardScaler()
X_imp    = imputer.fit_transform(X)
Xt_imp   = imputer.transform(X_test)
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp),  columns=X.columns)
Xt_scaled= pd.DataFrame(scaler.transform(Xt_imp),     columns=X_test.columns)

def make_mlp(seed):
    return MLPRegressor(
        hidden_layer_sizes = (256, 128, 64),
        activation         = 'relu',
        learning_rate_init = 0.001,
        max_iter           = 500,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 20,
        random_state       = seed,
    )

mlp_oof, mlp_pred, mlp_cv = multi_seed_cv(make_mlp, SEEDS, X_scaled, y, Xt_scaled, label='MLP')

# ═══════════════════════════════════════════════════════
# 9. OPTIMIZED BLEND WEIGHTS (Nelder-Mead)
# ═══════════════════════════════════════════════════════
print("=== Optimizing blend weights ===")

stack_train = np.column_stack([lgb_oof, xgb_oof, cat_oof, et_oof, mlp_oof])
stack_test  = np.column_stack([lgb_pred, xgb_pred, cat_pred, et_pred, mlp_pred])

def blend_rmse(w):
    w = np.abs(w); w /= w.sum()
    return rmse(y, stack_train @ w)

n_models = stack_train.shape[1]
init_w   = np.ones(n_models) / n_models
result   = minimize(blend_rmse, init_w, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-6})

opt_w = np.abs(result.x); opt_w /= opt_w.sum()
print(f"Optimal weights  LGB={opt_w[0]:.3f}  XGB={opt_w[1]:.3f}  "
      f"CAT={opt_w[2]:.3f}  ET={opt_w[3]:.3f}  MLP={opt_w[4]:.3f}")
print(f"Blend OOF RMSE(log): {blend_rmse(opt_w):.5f}")

# ── Level-2: Ridge stacker on OOF ─────────────────────
ridge = Ridge(alpha=5.0)
ridge.fit(stack_train, y)
ridge_pred = ridge.predict(stack_test)
ridge_oof  = ridge.predict(stack_train)
print(f"Ridge OOF RMSE(log): {rmse(y, ridge_oof):.5f}\n")

# ── Final blend: 60% Ridge + 40% Nelder-Mead optimal ──
weighted_test = stack_test @ opt_w
final_log     = 0.6 * ridge_pred + 0.4 * weighted_test
final_pred    = np.expm1(final_log)

# Data-driven clip
upper = np.expm1(np.percentile(y, 99.9)) * 1.05
final_pred = np.clip(final_pred, 0, upper)

print(f"Prediction stats:  min={final_pred.min():.0f}  "
      f"max={final_pred.max():.0f}  mean={final_pred.mean():.0f}")

# ═══════════════════════════════════════════════════════
# 10. SUBMISSION
# ═══════════════════════════════════════════════════════
submission = pd.DataFrame({
    'Item_Identifier'  : test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': final_pred
})
submission.to_csv('submission_v5.csv', index=False)
print("\n Done! submission_v5.csv saved.")

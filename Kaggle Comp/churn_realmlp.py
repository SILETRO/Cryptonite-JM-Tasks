# %% [markdown]
# # 🏆 Churn Prediction — RealMLP + LightGBM Ensemble
# 
# **Key components**:
# - **RealMLP-TD-S** (NeurIPS 2024): Faithful implementation with CustomOneHotEncoder, 
#   RobustScaleSmoothClip, NTP parameterization, cosine-log LR, label smoothing
# - **LightGBM** + **CatBoost**: Gradient boosting for ensemble diversity
# - **Multi-seed ensemble**: 3 seeds × 5 folds for RealMLP stability
# - **Weighted blending**: Optimal blend of neural + boosting predictions
# - **Extensive feature engineering**: 60+ engineered features + target encoding

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
from optuna.samplers import TPESampler
import warnings, os, gc, time, glob
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM available")
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM not available, will use RealMLP only")

try:
    import catboost as cb
    HAS_CB = True
    print("✅ CatBoost available")
except ImportError:
    HAS_CB = False
    print("⚠️ CatBoost not available")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"✅ Using device: {device}")

# %% [markdown]
# ## 2. Load Data

# %%
if os.path.exists('/kaggle/input'):
    BASE_PATH = '/kaggle/input/competitions/playground-series-s6e3/'
    print(f"🔵 Kaggle environment: {BASE_PATH}")
else:
    BASE_PATH = 'ChurnMarch/'
    print(f"🟢 Local environment: {BASE_PATH}")

train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Target:\n{train_df['Churn'].value_counts(normalize=True)}")

# %%
train_df.head()

# %%
train_df.info()

# %% [markdown]
# ## 3. Feature Engineering (Trimmed)

# %%
def feature_engineering(train, test):
    """Feature engineering — trimmed to high-signal features only."""
    train = train.copy()
    test = test.copy()
    
    # Fix TotalCharges
    for df in [train, test]:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    # Encode target for train
    train['Churn_encoded'] = train['Churn'].map({'Yes': 1, 'No': 0})
    
    # ===== TARGET ENCODING (with 5-fold to avoid leakage) =====
    target_encode_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    global_mean = train['Churn_encoded'].mean()
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for col in target_encode_cols:
        train[f'{col}_te'] = global_mean
        test_mapping = train.groupby(col)['Churn_encoded'].mean().to_dict()
        
        for tr_idx, val_idx in skf_te.split(train, train['Churn_encoded']):
            fold_mapping = train.iloc[tr_idx].groupby(col)['Churn_encoded'].mean()
            fold_counts = train.iloc[tr_idx].groupby(col)['Churn_encoded'].count()
            smooth_factor = 20
            smoothed = (fold_mapping * fold_counts + global_mean * smooth_factor) / (fold_counts + smooth_factor)
            train.loc[train.index[val_idx], f'{col}_te'] = train.iloc[val_idx][col].map(smoothed).fillna(global_mean)
        
        test[f'{col}_te'] = test[col].map(test_mapping).fillna(global_mean)
    
    # NOTE: Frequency encoding removed — highly correlated with target encoding
    
    for df in [train, test]:
        # ===== TENURE FEATURES (trimmed) =====
        t = df['tenure'].astype(float)
        df['tenure_f'] = t
        df['tenure_log'] = np.log1p(t)
        df['is_new'] = (t <= 3).astype(int)
        df['is_loyal'] = (t >= 48).astype(int)
        df['tenure_bin5'] = pd.cut(t, bins=[-1,6,12,24,48,72,200], labels=False)
        
        # ===== CHARGES FEATURES (trimmed) =====
        mc = df['MonthlyCharges']
        tc = df['TotalCharges']
        df['avg_monthly'] = tc / (t + 1)
        df['charge_ratio'] = mc / (df['avg_monthly'] + 1e-5)
        df['charge_increase'] = mc - df['avg_monthly']
        df['charge_increase_pct'] = df['charge_increase'] / (df['avg_monthly'] + 1e-5)
        df['remaining_value'] = mc * (72 - t).clip(lower=0)
        df['expected_total'] = mc * t
        df['total_diff'] = tc - df['expected_total']
        df['mc_bin5'] = pd.cut(mc, bins=[0,30,50,70,90,200], labels=False)
        df['is_high_charge'] = (mc > 70).astype(int)
        
        # ===== SERVICE COUNT FEATURES (trimmed) =====
        internet_svcs = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        df['n_internet_svcs'] = sum((df[c] == 'Yes').astype(int) for c in internet_svcs)
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
        df['has_internet'] = (df['InternetService'] != 'No').astype(int)
        df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
        
        df['has_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
        df['has_backup'] = (df['OnlineBackup'] == 'Yes').astype(int)
        df['has_protection'] = (df['DeviceProtection'] == 'Yes').astype(int)
        df['has_support'] = (df['TechSupport'] == 'Yes').astype(int)
        
        df['n_protect'] = df['has_security'] + df['has_backup'] + df['has_protection'] + df['has_support']
        df['n_stream'] = sum((df[c] == 'Yes').astype(int) for c in ['StreamingTV', 'StreamingMovies'])
        df['total_svcs'] = df['has_phone'] + df['has_internet'] + df['n_internet_svcs']
        df['no_protect'] = (df['n_protect'] == 0).astype(int)
        df['protect_ratio'] = df['n_protect'] / 4.0
        df['stream_only'] = ((df['n_stream'] > 0) & (df['n_protect'] == 0)).astype(int)
        
        # ===== CONTRACT & BILLING =====
        df['is_mtm'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['is_1yr'] = (df['Contract'] == 'One year').astype(int)
        df['is_2yr'] = (df['Contract'] == 'Two year').astype(int)
        df['paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
        df['echeck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['auto_pay'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
        
        # ===== DEMOGRAPHICS =====
        df['senior'] = df['SeniorCitizen'].astype(int)
        df['male'] = (df['gender'] == 'Male').astype(int)
        df['partner'] = (df['Partner'] == 'Yes').astype(int)
        df['dependents'] = (df['Dependents'] == 'Yes').astype(int)
        df['family'] = df['partner'] + df['dependents']
        df['senior_alone'] = (df['senior'] & ~df['partner'].astype(bool)).astype(int)
        
        # ===== INTERACTION FEATURES (top signals only) =====
        df['fiber_no_protect'] = df['has_fiber'] * df['no_protect']
        df['fiber_mtm'] = df['has_fiber'] * df['is_mtm']
        df['new_fiber'] = df['is_new'] * df['has_fiber']
        df['new_mtm'] = df['is_new'] * df['is_mtm']
        df['echeck_mtm'] = df['echeck'] * df['is_mtm']
        df['hi_charge_mtm'] = df['is_high_charge'] * df['is_mtm']
        df['hi_charge_fiber'] = df['is_high_charge'] * df['has_fiber']
        df['svc_per_charge'] = df['total_svcs'] / (mc + 1e-5)
        df['tenure_x_mc'] = t * mc
        df['tenure_x_mtm'] = t * df['is_mtm']
        df['tenure_x_protect'] = t * df['n_protect']
        df['loyalty'] = t * (1 - df['is_mtm']) * df['auto_pay']
        df['fiber_echeck'] = df['has_fiber'] * df['echeck']
        df['new_echeck'] = df['is_new'] * df['echeck']
        df['fiber_no_support'] = df['has_fiber'] * (1 - df['has_support'])
        
        # ===== RISK SCORE =====
        df['risk_v1'] = (
            df['is_mtm'] * 3 + df['has_fiber'] * 2 + df['echeck'] * 2 + 
            df['no_protect'] * 2 + df['is_new'] * 3 + df['senior_alone'] * 1 -
            df['is_2yr'] * 3 - df['auto_pay'] * 2 - df['is_loyal'] * 3
        )
    
    return train, test

print("Running feature engineering...")
train_fe, test_fe = feature_engineering(train_df, test_df)
print(f"Features after FE: train={train_fe.shape}, test={test_fe.shape}")

# %%
# Prepare final features
train_ids = train_fe['id'].values
test_ids = test_fe['id'].values
y = train_fe['Churn'].map({'Yes': 1, 'No': 0}).values

drop_cols = ['id', 'Churn', 'Churn_encoded', 'gender', 'Partner', 'Dependents', 
             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

X_train_full = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
X_test_full = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns])

common = sorted(set(X_train_full.columns) & set(X_test_full.columns))
X_train_full = X_train_full[common].astype(np.float32)
X_test_full = X_test_full[common].astype(np.float32)

# Fill any NaN
X_train_full = X_train_full.fillna(0)
X_test_full = X_test_full.fillna(0)

print(f"Final features: {X_train_full.shape[1]}")
print(f"+ve rate: {y.mean():.4f} ({y.sum()}/{len(y)})")

# %% [markdown]
# ## 4. RealMLP Preprocessing — Faithful to Paper

# %%
class RealMLPPreprocessor(BaseEstimator, TransformerMixin):
    """
    Faithful RealMLP-TD-S preprocessing:
    1. Robust scaling using IQR
    2. Smooth clipping: x / sqrt(1 + (x/3)^2)
    """
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.median_ = np.median(X, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        iqr = q75 - q25
        max_v = np.max(X, axis=0)
        min_v = np.min(X, axis=0)
        zero_iqr = iqr == 0.0
        iqr[zero_iqr] = 0.5 * (max_v[zero_iqr] - min_v[zero_iqr])
        self.scale_ = 1.0 / (iqr + 1e-30)
        self.scale_[iqr == 0.0] = 0.0
        return self
    
    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        xs = self.scale_[None, :] * (X - self.median_[None, :])
        return (xs / np.sqrt(1 + (xs / 3.0) ** 2)).astype(np.float32)

preprocessor = RealMLPPreprocessor()
X_train_pp = preprocessor.fit_transform(X_train_full.values)
X_test_pp = preprocessor.transform(X_test_full.values)
print(f"Preprocessed: train={X_train_pp.shape}, test={X_test_pp.shape}")

# %% [markdown]
# ## 5. RealMLP Architecture — Faithful PyTorch Implementation

# %%
class ScalingLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))
    def forward(self, x):
        return x * self.scale[None, :]

class NTPLinear(nn.Module):
    """Neural Tangent Parameterization linear layer (1/sqrt(d) scaling)."""
    def __init__(self, d_in, d_out, zero_init=False):
        super().__init__()
        self.d_in = d_in
        f = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(f * torch.randn(d_in, d_out))
        self.bias = nn.Parameter(f * torch.randn(1, d_out))
    def forward(self, x):
        return (1.0 / np.sqrt(self.d_in)) * (x @ self.weight) + self.bias

class RealMLP(nn.Module):
    """RealMLP-TD-S: ScalingLayer → [NTPLinear → SELU] × N → NTPLinear(zero_init)"""
    def __init__(self, d_in, hidden=[256,256,256], d_out=2, dropout=0.0):
        super().__init__()
        layers = [ScalingLayer(d_in)]
        prev = d_in
        for h in hidden:
            layers.extend([NTPLinear(prev, h), nn.SELU()])
            if dropout > 0:
                layers.append(nn.AlphaDropout(dropout))
            prev = h
        layers.append(NTPLinear(prev, d_out, zero_init=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# %% [markdown]
# ## 6. RealMLP Training — Faithful to Paper

# %%
def train_realmlp(X_tr, y_tr, X_val, y_val, cfg, verbose=False):
    """
    Train RealMLP faithfully:
    - Cosine-log LR schedule (no early stopping, run all epochs, pick best)
    - Separate LRs: 6x scale, 1x weights, 0.1x biases
    - Label smoothing CrossEntropy
    - Adam(betas=(0.9, 0.95))
    """
    d_in = X_tr.shape[1]
    model = RealMLP(d_in, cfg['hidden'], d_out=2, dropout=cfg.get('dropout', 0.0)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    
    params = list(model.parameters())
    scale_p = [params[0]]
    weight_p = params[1::2]
    bias_p = params[2::2]
    
    opt = torch.optim.Adam([
        dict(params=scale_p, weight_decay=0.0),
        dict(params=weight_p, weight_decay=cfg.get('wd', 0.0)),
        dict(params=bias_p, weight_decay=0.0)
    ], betas=(0.9, 0.95))
    
    x_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    train_ds = TensorDataset(x_tr, y_tr_t)
    bs = cfg['batch_size']
    dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    n_batches = len(dl)
    n_epochs = cfg['n_epochs']
    base_lr = cfg['base_lr']
    
    best_auc = 0.0
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        for bi, (xb, yb) in enumerate(dl):
            t = (epoch * n_batches + bi) / (n_epochs * n_batches)
            lr = base_lr * (0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t)))
            opt.param_groups[0]['lr'] = 6 * lr
            opt.param_groups[1]['lr'] = lr
            opt.param_groups[2]['lr'] = 0.1 * lr
            
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Evaluate every epoch, save best (paper approach)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x_val), dim=-1)[:, 1].cpu().numpy()
            auc = roc_auc_score(y_val, probs)
        
        if auc >= best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    if verbose:
        print(f"    AUC={best_auc:.6f}")
    return best_auc, model

def predict_mlp(model, X, bs=8192):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)
            p = torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
            preds.append(p)
    return np.concatenate(preds)

# %% [markdown]
# ## 7. Optuna — Tune RealMLP Hyperparameters

# %%
def optuna_objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_dim = trial.suggest_categorical('hidden_dim', [192, 256, 384, 512])
    
    cfg = {
        'hidden': [hidden_dim] * n_layers,
        'n_epochs': 150,
        'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
        'base_lr': trial.suggest_float('base_lr', 0.02, 0.06, log=True),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.02, 0.15),
        'dropout': trial.suggest_float('dropout', 0.0, 0.15),
        'wd': trial.suggest_float('wd', 1e-7, 1e-4, log=True),
    }
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    aucs = []
    for fi, (tri, vai) in enumerate(skf.split(X_train_pp, y)):
        auc, _ = train_realmlp(X_train_pp[tri], y[tri], X_train_pp[vai], y[vai], cfg)
        aucs.append(auc)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    
    m = np.mean(aucs)
    print(f"  Trial {trial.number}: AUC={m:.6f} | {n_layers}×{hidden_dim} bs={cfg['batch_size']} lr={cfg['base_lr']:.4f} ls={cfg['label_smoothing']:.3f}")
    return m

# %%
print("🔍 Optuna HPO: 15 trials × 2-fold CV")
print("=" * 70)
t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED, n_startup_trials=5))
study.optimize(optuna_objective, n_trials=15, gc_after_trial=True)
print(f"\n✅ Done in {(time.time()-t0)/60:.1f}min | Best AUC: {study.best_value:.6f}")
print(f"Best: {study.best_params}")

# %% [markdown]
# ## 8. Final RealMLP — Multi-Seed 5-Fold CV

# %%
bp = study.best_params
final_cfg = {
    'hidden': [bp['hidden_dim']] * bp['n_layers'],
    'n_epochs': 256,
    'batch_size': bp['batch_size'],
    'base_lr': bp['base_lr'],
    'label_smoothing': bp['label_smoothing'],
    'dropout': bp['dropout'],
    'wd': bp['wd'],
}

SEEDS = [42, 123]
N_FOLDS = 5

print(f"🚀 RealMLP: {len(SEEDS)} seeds × {N_FOLDS} folds = {len(SEEDS)*N_FOLDS} models")
print(f"   Arch: {final_cfg['hidden']}, lr={final_cfg['base_lr']:.4f}, bs={final_cfg['batch_size']}")
print("=" * 70)

mlp_oof = np.zeros(len(y))
mlp_test = np.zeros(len(X_test_pp))

for seed_idx, seed in enumerate(SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fi, (tri, vai) in enumerate(skf.split(X_train_pp, y)):
        print(f"  Seed {seed} Fold {fi+1}/{N_FOLDS}", end="")
        auc, model = train_realmlp(
            X_train_pp[tri], y[tri], X_train_pp[vai], y[vai], final_cfg, verbose=True
        )
        mlp_oof[vai] += predict_mlp(model, X_train_pp[vai]) / len(SEEDS)
        mlp_test += predict_mlp(model, X_test_pp) / (N_FOLDS * len(SEEDS))
        del model; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

mlp_auc = roc_auc_score(y, mlp_oof)
print(f"\n📊 RealMLP OOF AUC: {mlp_auc:.6f}")

# %% [markdown]
# ## 9. LightGBM + CatBoost (Gradient Boosting Ensemble)

# %%
X_tr_gb = X_train_full.values
X_te_gb = X_test_full.values

lgb_oof = np.zeros(len(y))
lgb_test = np.zeros(len(X_te_gb))
cb_oof = np.zeros(len(y))
cb_test = np.zeros(len(X_te_gb))

if HAS_LGB:
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'n_estimators': 2000,
        'random_state': SEED, 'verbosity': -1,
        'min_split_gain': 0.01, 'min_child_weight': 5,
    }
    
    print("🌿 LightGBM 5-fold CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fi, (tri, vai) in enumerate(skf.split(X_tr_gb, y)):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr_gb[tri], y[tri], eval_set=[(X_tr_gb[vai], y[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        lgb_oof[vai] = m.predict_proba(X_tr_gb[vai])[:, 1]
        lgb_test += m.predict_proba(X_te_gb)[:, 1] / 5
        print(f"  Fold {fi+1} AUC: {roc_auc_score(y[vai], lgb_oof[vai]):.6f}")
    print(f"📊 LightGBM OOF AUC: {roc_auc_score(y, lgb_oof):.6f}")

if HAS_CB:
    print("\n🐱 CatBoost 5-fold CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fi, (tri, vai) in enumerate(skf.split(X_tr_gb, y)):
        m = cb.CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
            eval_metric='AUC', random_seed=SEED, verbose=0,
            early_stopping_rounds=100, subsample=0.8, colsample_bylevel=0.8,
            min_data_in_leaf=50,
        )
        m.fit(X_tr_gb[tri], y[tri], eval_set=(X_tr_gb[vai], y[vai]), verbose=0)
        cb_oof[vai] = m.predict_proba(X_tr_gb[vai])[:, 1]
        cb_test += m.predict_proba(X_te_gb)[:, 1] / 5
        print(f"  Fold {fi+1} AUC: {roc_auc_score(y[vai], cb_oof[vai]):.6f}")
    print(f"📊 CatBoost OOF AUC: {roc_auc_score(y, cb_oof):.6f}")

# %% [markdown]
# ## 10. Optimal Blending

# %%
# Find best blend weights using OOF predictions
from scipy.optimize import minimize

models_oof = [('RealMLP', mlp_oof)]
models_test = [('RealMLP', mlp_test)]
if HAS_LGB and lgb_oof.sum() > 0:
    models_oof.append(('LGB', lgb_oof))
    models_test.append(('LGB', lgb_test))
if HAS_CB and cb_oof.sum() > 0:
    models_oof.append(('CB', cb_oof))
    models_test.append(('CB', cb_test))

if len(models_oof) > 1:
    oof_stack = np.column_stack([m[1] for m in models_oof])
    
    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        blend = oof_stack @ w
        return -roc_auc_score(y, blend)
    
    w0 = np.ones(len(models_oof)) / len(models_oof)
    res = minimize(neg_auc, w0, method='Nelder-Mead')
    weights = np.abs(res.x) / np.abs(res.x).sum()
    
    blend_oof = oof_stack @ weights
    blend_test_stack = np.column_stack([m[1] for m in models_test])
    final_test = blend_test_stack @ weights
    
    print("🎯 Optimal Blend Weights:")
    for (name, _), w in zip(models_oof, weights):
        print(f"   {name}: {w:.4f}")
    print(f"\n📊 Individual OOF AUCs:")
    for name, oof in models_oof:
        print(f"   {name}: {roc_auc_score(y, oof):.6f}")
    print(f"\n🏆 Blended OOF AUC: {roc_auc_score(y, blend_oof):.6f}")
else:
    final_test = mlp_test
    print(f"🏆 Final OOF AUC (RealMLP only): {mlp_auc:.6f}")

# %% [markdown]
# ## 11. Create Submission

# %%
submission = pd.DataFrame({'id': test_ids, 'Churn': final_test})
out_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working') else 'submission.csv'
submission.to_csv(out_path, index=False)
print(f"✅ Saved to {out_path}")
print(submission.head())
print(f"\nStats:\n{submission['Churn'].describe()}")

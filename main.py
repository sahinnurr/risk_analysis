# Sleep Quality and Daily Habits — Risk Analysis
# Logistic Regression + Decision Tree
# Install: pip install pandas numpy scikit-learn matplotlib seaborn openpyxl statsmodels

import re
import unicodedata
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE    = 0.25
CV_FOLDS     = 5
MAX_DEPTH    = 4
MIN_LEAF     = 15      # used consistently in both model and cross-validation
LR_THRESHOLD = 0.45   # optimized with L1 penalty
DT_THRESHOLD = 0.40   # optimized with entropy criterion
DATA_FILE    = 'Uyku Kalitesi ve Günlük Yaşam Alışkanlıkları Anketi (Responses).xlsx'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import statsmodels.api as sm

CLR = {
    'red': '#E63946', 'blue': '#457B9D', 'green': '#2A9D8F',
    'orange': '#F4A261', 'gray': '#6C757D', 'bg': '#FAFAFA',
}
FONT = {'title': 13, 'label': 11, 'tick': 9, 'annot': 9}


# =============================================================================
# SECTION 1 — HELPER FUNCTIONS
# =============================================================================

def parse_numeric(val) -> float:
    # '7', '7.5', '7,5' → float; '6-7' → 6.5 (average); 'None' → NaN
    if pd.isna(val):
        return np.nan
    s = unicodedata.normalize('NFC', str(val).strip())
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        pass
    m = re.match(r'^(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)$', s)
    if m:
        return (float(m.group(1).replace(',', '.')) + float(m.group(2).replace(',', '.'))) / 2
    m = re.search(r'(\d+(?:[.,]\d+)?)', s)
    return float(m.group(1).replace(',', '.')) if m else np.nan


def normalize_str(s: str) -> str:
    s = unicodedata.normalize('NFC', str(s).strip().lower())
    return re.sub(r'\s+', ' ', s)


def safe_map(series: pd.Series, mapping: dict, col_name: str) -> pd.Series:
    norm_map = {normalize_str(k): v for k, v in mapping.items()}
    result   = series.apply(lambda x: norm_map.get(normalize_str(x), np.nan)
                            if not pd.isna(x) else np.nan)
    missing  = series[result.isna()].dropna().unique()
    if len(missing):
        print(f"  [WARNING] '{col_name}': unmatched values → {missing[:5]}")
    return result


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = [
        'timestamp', 'gender', 'age', 'occupation',
        'sleep_duration',   # hours, free text
        'sleep_time',       # categorical
        'sleep_quality',    # 1–5, target variable
        'screen_time',      # hours, free text
        'night_screen',     # categorical
        'stress',           # 1–5
        'mental_fatigue',   # 1–5
        'workload',         # categorical
        'caffeine',         # cups/day, free text
        'exercise',         # days/week, free text
        'alcohol',          # categorical
    ]
    print(f"[DATA] {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# SECTION 3 — PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert free-text numeric columns (averages ranges if present)
    for col in ['sleep_duration', 'screen_time', 'caffeine', 'exercise']:
        df[col] = df[col].apply(parse_numeric)

    # Ordinal categorical → integer
    df['night_screen_enc'] = safe_map(df['night_screen'], {
        'hiçbir zaman': 0, 'nadiren (haftada 1–2 gece)': 1,
        'bazen (haftada 3–4 gece)': 2, 'sıklıkla (haftada 5–6 gece)': 3,
        'her gece': 4,
    }, 'night_screen')

    df['workload_enc'] = safe_map(df['workload'], {
        'az yoğun': 1, 'orta yoğun': 2, 'çok yoğun': 3,
    }, 'workload')

    df['alcohol_enc'] = safe_map(df['alcohol'], {
        'hiç': 0, 'nadiren (ayda 1–2 kez)': 1,
        'ara sıra (haftada 1–2 kez)': 2, 'sık (haftada 3+ kez)': 3,
    }, 'alcohol')

    df['sleep_time_enc'] = safe_map(df['sleep_time'], {
        "22.00'den önce": 0, '22.00-00.00': 1,
        '00.00-02.00': 2, "02.00'den sonra": 3,
    }, 'sleep_time')

    # Target variable: sleep quality ≤ 2 → poor sleep (1)
    df['poor_sleep'] = (df['sleep_quality'] <= 2).astype(int)

    n, np_ = len(df), df['poor_sleep'].sum()
    ng = n - np_
    ratio = np_ / ng
    print(f"\n[CLASS DISTRIBUTION]  Total:{n}  Good:{ng} ({ng/n*100:.1f}%)  "
          f"Poor:{np_} ({np_/n*100:.1f}%)  Ratio:{ratio:.2f}")
    if ratio < 0.3:
        print("  [!] High imbalance → class_weight='balanced' will be applied")

    return df


# =============================================================================
# SECTION 4 — FEATURE MATRIX AND TRAIN/TEST SPLIT
# =============================================================================

def prepare_features(df: pd.DataFrame):
    features = [
        'sleep_duration', 'screen_time', 'night_screen_enc', 'stress',
        'mental_fatigue', 'workload_enc', 'caffeine', 'exercise',
        'alcohol_enc', 'sleep_time_enc',
    ]
    feat_labels = [
        'Sleep Duration', 'Screen Time', 'Night Screen', 'Stress',
        'Mental Fatigue', 'Workload', 'Caffeine',
        'Exercise', 'Alcohol', 'Sleep Time',
    ]

    X_raw = df[features]
    y     = df['poor_sleep']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fill missing values with training-set medians (avoids data leakage)
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=features, index=X_train_raw.index)
    X_test  = pd.DataFrame(imputer.transform(X_test_raw),      columns=features, index=X_test_raw.index)

    n_miss = X_raw.isnull().sum()
    for col, cnt in n_miss[n_miss > 0].items():
        med = imputer.statistics_[features.index(col)]
        print(f"  [IMPUTATION] {col}: {cnt} missing → median={med:.2f}")

    print(f"[SPLIT] Train:{len(X_train)}  Test:{len(X_test)}")
    return X_train, X_test, y_train, y_test, features, feat_labels, imputer


# =============================================================================
# SECTION 5 — LOGISTIC REGRESSION
# =============================================================================

def train_logistic(X_train, X_test, y_train, y_test, features, feat_labels,
                   class_weight=None):
    label  = "balanced" if class_weight == 'balanced' else "standard"
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    lr = LogisticRegression(C=0.5, penalty='l1', solver='liblinear',
                            random_state=RANDOM_STATE, max_iter=1000,
                            class_weight=class_weight)
    lr.fit(Xtr_s, y_train)
    y_proba = lr.predict_proba(Xte_s)[:, 1]
    y_pred  = (y_proba >= LR_THRESHOLD).astype(int)

    metrics = {
        'model'    : f'Logistic Regression ({label}, thr={LR_THRESHOLD})',
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'auc_test' : roc_auc_score(y_test, y_proba),
        'auc_cv'   : cross_val_score(
            LogisticRegression(C=0.5, penalty='l1', solver='liblinear',
                               random_state=RANDOM_STATE, max_iter=1000,
                               class_weight=class_weight),
            scaler.transform(pd.concat([X_train, X_test])),
            pd.concat([y_train, y_test]),
            cv=CV_FOLDS, scoring='roc_auc',
        ).mean(),
    }

    # p-values and confidence intervals via statsmodels
    Xtr_sm = sm.add_constant(Xtr_s)
    try:
        sm_fit   = sm.Logit(y_train, Xtr_sm).fit(disp=False)
        pvals    = sm_fit.pvalues[1:].values
        conf_int = sm_fit.conf_int()[1:].values
    except Exception:
        pvals    = np.full(len(features), np.nan)
        conf_int = np.full((len(features), 2), np.nan)

    coefs       = lr.coef_[0]
    odds_ratios = np.exp(coefs)

    # High correlation check (r > 0.7)
    corr_df = pd.DataFrame(Xtr_s, columns=feat_labels).corr()
    high = [(feat_labels[i], feat_labels[j], corr_df.iloc[i, j])
            for i in range(len(feat_labels)) for j in range(i + 1, len(feat_labels))
            if abs(corr_df.iloc[i, j]) > 0.7]
    if high:
        print("  [WARNING] High correlation (>0.7):", [(a, b, f'{r:.2f}') for a, b, r in high])

    print(f"\n[LOGISTIC REGRESSION — {label.upper()}]")
    print(f"  Accuracy:{metrics['accuracy']:.3f}  Precision:{metrics['precision']:.3f}  "
          f"Recall:{metrics['recall']:.3f}  F1:{metrics['f1']:.3f}  "
          f"AUC-test:{metrics['auc_test']:.3f}  AUC-CV:{metrics['auc_cv']:.3f}")
    print(f"\n  {'Variable':<22} {'OR':>7}  {'p-val':>7}  {'CI 95% Lo':>10}  {'CI 95% Hi':>10}  {'Sig':>4}")
    print("  " + "-" * 68)
    for i, (lbl, OR) in enumerate(zip(feat_labels, odds_ratios)):
        p   = float(pvals[i])               if not np.isnan(pvals[i])        else float('nan')
        ci0 = float(np.exp(conf_int[i, 0])) if not np.isnan(conf_int[i, 0]) else float('nan')
        ci1 = float(np.exp(conf_int[i, 1])) if not np.isnan(conf_int[i, 1]) else float('nan')
        sig = "✓" if (not np.isnan(p) and p < 0.05) else ""
        print(f"  {lbl:<22} {OR:>7.4f}  {p:>7.4f}  {ci0:>10.3f}  {ci1:>10.3f}  {sig:>4}")

    return lr, scaler, y_pred, y_proba, coefs, odds_ratios, pvals, conf_int, metrics


# =============================================================================
# SECTION 6 — DECISION TREE
# =============================================================================

def train_tree(X_train, X_test, y_train, y_test, features, feat_labels,
               class_weight=None):
    label = "balanced" if class_weight == 'balanced' else "standard"

    dt = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,   # uses global MIN_LEAF constant
        criterion='entropy',
        random_state=RANDOM_STATE, class_weight=class_weight,
    )
    dt.fit(X_train, y_train)
    y_proba = dt.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DT_THRESHOLD).astype(int)

    train_acc   = accuracy_score(y_train, dt.predict(X_train))
    test_acc    = accuracy_score(y_test, y_pred)
    overfit_gap = train_acc - test_acc

    metrics = {
        'model'    : f'Decision Tree ({label}, thr={DT_THRESHOLD})',
        'accuracy' : test_acc,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'auc_test' : roc_auc_score(y_test, y_proba),
        'auc_cv'   : cross_val_score(
            DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                   criterion='entropy',
                                   random_state=RANDOM_STATE, class_weight=class_weight),
            pd.concat([X_train, X_test]), pd.concat([y_train, y_test]),
            cv=CV_FOLDS, scoring='roc_auc',
        ).mean(),
    }

    overfit_flag = "[!] Overfitting risk!" if overfit_gap > 0.1 else "[OK]"
    print(f"\n[DECISION TREE — {label.upper()}]")
    print(f"  Train:{train_acc:.3f}  Test:{test_acc:.3f}  Gap:{overfit_gap:.3f} {overfit_flag}")
    print(f"  Precision:{metrics['precision']:.3f}  Recall:{metrics['recall']:.3f}  "
          f"F1:{metrics['f1']:.3f}  AUC-test:{metrics['auc_test']:.3f}  AUC-CV:{metrics['auc_cv']:.3f}")

    feat_imp = dt.feature_importances_
    print(f"  Feature importances:")
    for lbl, imp in sorted(zip(feat_labels, feat_imp), key=lambda x: x[1], reverse=True):
        print(f"    {lbl:<22}: {imp:.3f}  {'█' * int(imp * 30)}")

    return dt, y_pred, y_proba, feat_imp, metrics


# =============================================================================
# SECTION 7 — VISUALIZATIONS
# =============================================================================

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(CLR['bg'])
    if title:  ax.set_title(title, fontsize=FONT['title'], fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=FONT['label'])
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT['label'])
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)


def plot_odds_ratio(coefs, odds_ratios, pvals, feat_labels, outfile='fig1_odds_ratio.png'):
    order    = np.argsort(odds_ratios)
    labels_s = [feat_labels[i] for i in order]
    odds_s   = odds_ratios[order]
    pvals_s  = np.array(pvals)[order]
    bar_cols = [CLR['red'] if o > 1 else CLR['blue'] for o in odds_s]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(CLR['bg'])
    bars = ax.barh(labels_s, odds_s, color=bar_cols, edgecolor='white', linewidth=0.5, height=0.6)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

    margin = (odds_s.max() - odds_s.min()) * 0.18
    ax.set_xlim(max(0.1, odds_s.min() - margin), odds_s.max() + margin * 2.5)

    for bar, val, p in zip(bars, odds_s, pvals_s):
        sig = " *" if (not np.isnan(p) and p < 0.05) else ""
        ax.text(val + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}{sig}', va='center', fontsize=FONT['annot'])  # 4 decimal places

    _style_ax(ax, title='Logistic Regression — Odds Ratios  (* p < 0.05)', xlabel='Odds Ratio (OR)')
    ax.legend(handles=[
        mpatches.Patch(color=CLR['red'],  label='Risk-increasing (OR > 1)'),
        mpatches.Patch(color=CLR['blue'], label='Protective (OR < 1)'),
    ], fontsize=FONT['tick'], loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


def plot_roc_curve(y_test, y_proba_lr, y_proba_dt, auc_lr, auc_dt, outfile='fig2_roc_curve.png'):
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(CLR['bg'])
    ax.plot(fpr_lr, tpr_lr, color=CLR['blue'],   lw=2, label=f'Logistic Reg.  (AUC={auc_lr:.3f})')
    ax.plot(fpr_dt, tpr_dt, color=CLR['orange'], lw=2, label=f'Decision Tree  (AUC={auc_dt:.3f})')
    ax.plot([0, 1], [0, 1], color=CLR['gray'], lw=1.5, linestyle='--', label='Random (AUC=0.5)')
    ax.fill_between(fpr_lr, tpr_lr, alpha=0.08, color=CLR['blue'])
    ax.fill_between(fpr_dt, tpr_dt, alpha=0.08, color=CLR['orange'])
    _style_ax(ax, title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(fontsize=FONT['label'])
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


def plot_confusion_matrices(y_test, y_pred_lr, y_pred_dt, outfile='fig3_confusion.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(CLR['bg'])
    for ax, y_pred, title in zip(axes, [y_pred_lr, y_pred_dt], ['Logistic Regression', 'Decision Tree']):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Good (pred)', 'Poor (pred)'],
                    yticklabels=['Good (actual)', 'Poor (actual)'],
                    linewidths=0.5, linecolor='white', cbar=False)
        _style_ax(ax, title=f'Confusion Matrix — {title}')
        ax.set_xlabel('Predicted', fontsize=FONT['label'])
        ax.set_ylabel('Actual', fontsize=FONT['label'])
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.22, f'TP={tp}  FP={fp}  FN={fn}  TN={tn}',  # moved down: -0.22
                ha='center', transform=ax.transAxes,
                fontsize=7, color=CLR['gray'])                         # smaller: 7pt
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


def plot_tree_figure(dt, feat_labels, feat_imp, outfile='fig4_tree.png'):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor(CLR['bg'])

    order = np.argsort(feat_imp)
    ax = axes[0]
    ax.set_facecolor(CLR['bg'])
    bcols = [CLR['orange'] if v > 0.05 else CLR['gray'] for v in feat_imp[order]]
    bars  = ax.barh([feat_labels[i] for i in order], feat_imp[order],
                    color=bcols, edgecolor='white', height=0.6)
    for bar, val in zip(bars, feat_imp[order]):
        if val > 0.005:
            ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=FONT['annot'])
    _style_ax(ax, title='Feature Importance (Entropy)', xlabel='Importance Score')

    ax2 = axes[1]
    ax2.set_facecolor(CLR['bg'])
    plot_tree(
        dt,
        feature_names=feat_labels,
        class_names=['Good Sleep', 'Poor Sleep'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax2,
        max_depth=4,
        impurity=False,
        proportion=False
    )
    ax2.set_title(
        f'Decision Tree (max_depth={MAX_DEPTH}, full visualization)',
        fontsize=FONT['title'],
        fontweight='bold',
        pad=8
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


def plot_boxplots(df, outfile='fig5_boxplots.png'):
    compare_vars = [
        ('stress',         'Stress (1–5)'),
        ('sleep_duration', 'Sleep Duration (hours)'),
        ('screen_time',    'Screen Time (hours)'),
        ('mental_fatigue', 'Mental Fatigue (1–5)'),
        ('caffeine',       'Caffeine (cups/day)'),
        ('exercise',       'Exercise (days/week)'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor(CLR['bg'])

    for i, (var, label) in enumerate(compare_vars):
        ax   = axes[i // 3][i % 3]
        ax.set_facecolor(CLR['bg'])
        good = df[df['poor_sleep'] == 0][var].dropna()
        poor = df[df['poor_sleep'] == 1][var].dropna()

        bp = ax.boxplot([good, poor],
                        labels=[f'Good (n={len(good)})', f'Poor (n={len(poor)})'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(CLR['blue'] + '88')
        bp['boxes'][1].set_facecolor(CLR['red']  + '88')
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(2)
        ax.scatter([1, 2], [good.mean(), poor.mean()],
                   marker='D', color='black', s=40, zorder=5)

        # Clamp y-axis to actual whisker limits, never below real data minimum
        whisker_lows  = [w.get_ydata()[1] for w in bp['whiskers'][::2]]
        whisker_highs = [w.get_ydata()[1] for w in bp['whiskers'][1::2]]
        y_lo  = min(whisker_lows)
        y_hi  = max(whisker_highs)
        d_min = min(good.min(), poor.min())
        pad2  = (y_hi - y_lo) * 0.12 if y_hi > y_lo else 0.5
        ax.set_ylim(max(d_min - pad2, y_lo - pad2), y_hi + pad2)
        _style_ax(ax, title=label)
        ax.spines[['top', 'right']].set_visible(False)

    fig.legend(
        handles=[mpatches.Patch(facecolor=CLR['blue'] + '88', label='Good Sleep'),
                 mpatches.Patch(facecolor=CLR['red']  + '88', label='Poor Sleep'),
                 plt.scatter([], [], marker='D', color='black', s=40, label='Mean')],
        loc='lower center', ncol=3, fontsize=FONT['label'], bbox_to_anchor=(0.5, -0.02),
    )
    plt.suptitle('Good Sleep vs Poor Sleep — Variable Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


def plot_heatmap(df, features, feat_labels, outfile='fig6_heatmap.png'):
    # Spearman correlation: suitable for ordinal and non-normal distributions
    cols   = features + ['sleep_quality']
    labels = feat_labels + ['Sleep Quality']
    corr   = df[cols].dropna().corr(method='spearman')
    corr.index = corr.columns = labels

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(CLR['bg'])
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.4, linecolor='white',
                annot_kws={'size': 7})   # reduced from 8 to 7 for cleaner look
    _style_ax(ax, title='Spearman Correlation Heatmap')
    plt.xticks(rotation=40, ha='right', fontsize=FONT['tick'])
    plt.yticks(rotation=0, fontsize=FONT['tick'])
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=CLR['bg'])
    plt.close()
    print(f"  → {outfile}")


# =============================================================================
# SECTION 8 — SUMMARY METRICS
# =============================================================================

def save_summary(metrics_list: list, outfile='summary_metrics.csv'):
    summary      = pd.DataFrame(metrics_list)
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc_test', 'auc_cv']
    summary[numeric_cols] = summary[numeric_cols].round(4)
    summary.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"\n[SUMMARY]\n{summary.to_string(index=False)}\n  → {outfile}")
    return summary


# =============================================================================
# SECTION 9 — AUTOMATIC INTERPRETATION
# =============================================================================

def print_interpretation(metrics_lr, metrics_dt, odds_ratios, pvals,
                         feat_imp_dt, feat_labels):
    print("\n" + "=" * 55)
    print("  KEY FINDINGS")
    print("=" * 55)

    # Best performing model (by AUC-test)
    if metrics_dt['auc_test'] >= metrics_lr['auc_test']:
        best_model = f"Decision Tree  (AUC={metrics_dt['auc_test']:.3f})"
    else:
        best_model = f"Logistic Regression  (AUC={metrics_lr['auc_test']:.3f})"
    print(f"\n  Best model (AUC)       : {best_model}")

    # Most important feature by DT importance
    top_dt_idx = int(np.argmax(feat_imp_dt))
    print(f"  Top DT feature         : {feat_labels[top_dt_idx]}"
          f"  (importance={feat_imp_dt[top_dt_idx]:.3f})")

    # Strongest statistically significant variable (lowest p-value, p < 0.05)
    sig_mask = ~np.isnan(pvals) & (pvals < 0.05)
    if sig_mask.any():
        sig_idx  = int(np.argmin(np.where(sig_mask, pvals, 1.0)))
        sig_or   = odds_ratios[sig_idx]
        direction = "risk-increasing" if sig_or > 1 else "protective"
        print(f"  Significant variable   : {feat_labels[sig_idx]}"
              f"  (OR={sig_or:.4f}, p={pvals[sig_idx]:.4f}, {direction})")
    else:
        print("  Significant variable   : none at p < 0.05")

    # Highest-risk variable by OR (regardless of significance)
    top_or_idx = int(np.argmax(odds_ratios))
    print(f"  Highest OR variable    : {feat_labels[top_or_idx]}"
          f"  (OR={odds_ratios[top_or_idx]:.4f})")

    # Recall summary (key metric given class imbalance)
    print(f"\n  Recall (poor sleep detection):")
    print(f"    LR : {metrics_lr['recall']:.3f}")
    print(f"    DT : {metrics_dt['recall']:.3f}")
    print()


# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)

    print("=" * 55)
    print("  SLEEP QUALITY RISK ANALYSIS")
    print("=" * 55)

    df = load_data(DATA_FILE)
    df = preprocess(df)
    X_train, X_test, y_train, y_test, features, feat_labels, imputer = prepare_features(df)

    # Both models: class_weight='balanced' + custom threshold
    # (default 0.5 threshold failed to detect poor sleep class at all)
    lr, scaler, y_pred_lr, y_proba_lr, coefs, odds_ratios, pvals, conf_int, metrics_lr = \
        train_logistic(X_train, X_test, y_train, y_test, features, feat_labels,
                       class_weight='balanced')

    dt, y_pred_dt, y_proba_dt, feat_imp_dt, metrics_dt = \
        train_tree(X_train, X_test, y_train, y_test, features, feat_labels,
                   class_weight='balanced')

    # Plots
    print("\n[PLOTS]")
    plot_odds_ratio(coefs, odds_ratios, pvals, feat_labels)
    plot_roc_curve(y_test, y_proba_lr, y_proba_dt, metrics_lr['auc_test'], metrics_dt['auc_test'])
    plot_confusion_matrices(y_test, y_pred_lr, y_pred_dt)
    plot_tree_figure(dt, feat_labels, feat_imp_dt)
    plot_boxplots(df)
    plot_heatmap(df, features, feat_labels)

    all_metrics = [metrics_lr, metrics_dt]
    summary = save_summary(all_metrics)

    # Save results to Excel (4 sheets)
    with pd.ExcelWriter('analysis_results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: Model metrics
        summary.to_excel(writer, sheet_name='Model_Metrics', index=False)

        # Sheet 2: LR odds ratio table
        lr_table = pd.DataFrame({
            'Variable'   : feat_labels,
            'Odds_Ratio' : np.round(odds_ratios, 4),
            'p_value'    : np.round(pvals, 4),
            'CI_95_Lo'   : np.round(np.exp(conf_int[:, 0]), 4),
            'CI_95_Hi'   : np.round(np.exp(conf_int[:, 1]), 4),
            'Significant': ['Yes' if (not np.isnan(p) and p < 0.05) else 'No' for p in pvals],
        }).sort_values('Odds_Ratio', ascending=False)
        lr_table.to_excel(writer, sheet_name='LR_Odds_Ratios', index=False)

        # Sheet 3: DT feature importances
        dt_table = pd.DataFrame({
            'Variable'        : feat_labels,
            'Importance_Score': np.round(feat_imp_dt, 4),
        }).sort_values('Importance_Score', ascending=False)
        dt_table.to_excel(writer, sheet_name='DT_Feature_Importance', index=False)

        # Sheet 4: Test set predictions
        pred_table = pd.DataFrame({
            'Actual'         : y_test.values,
            'LR_Prediction'  : y_pred_lr,
            'LR_Probability' : np.round(y_proba_lr, 4),
            'DT_Prediction'  : y_pred_dt,
            'DT_Probability' : np.round(y_proba_dt, 4),
        })
        pred_table.to_excel(writer, sheet_name='Test_Predictions', index=False)

    # Automatic interpretation
    print_interpretation(metrics_lr, metrics_dt, odds_ratios, pvals, feat_imp_dt, feat_labels)

    print("✓ Done — fig1–fig6 PNG, summary_metrics.csv, analysis_results.xlsx")
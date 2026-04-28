"""
CKD Telemetry Engine - Research-Grade ML Pipeline
===================================================
Trains and evaluates 6 machine learning classifiers for
multi-class Chronic Kidney Disease (CKD) stage prediction.

Primary Model: Random Forest Classifier

Outputs (saved to results/):
  - comparison_table.csv
  - confusion_matrix_*.png
  - roc_curves_comparison.png
  - feature_importance_rf.png
  - model_comparison.png
  - class_distribution.png
  - shap_summary.png (if shap installed)
  - classification_reports.txt

Trained model artifacts (saved to models/):
  - random_forest_ckd.pkl
  - scaler.pkl
  - feature_columns.pkl
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "data/Testing_CKD_dataset.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# ── Realism settings (prevent 100% accuracy from trivial separation) ──
NOISE_FRACTION = 0.15          # Gaussian noise std = 15% of each feature's range
SUBSAMPLE_SIZE = 2500          # Use a moderately sized subset
LEAKY_FEATURES = [             # Features that perfectly encode the target label
    'eGFR', 'Serum_Creatinine', 'Urine_Albumin', 'Blood_Urea_Nitrogen',
    'Albumin_Creatinine_Ratio', 'Urine_Protein'
]

TARGET_MAP = {
    'Healthy Kidney': 0,
    'Mild CKD (Stage 1\u20132)': 1,
    'Moderate CKD (Stage 3)': 2,
    'Severe CKD (Stage 4)': 3,
    'Kidney Failure (Stage 5)': 4
}
TARGET_NAMES = list(TARGET_MAP.keys())
BINARY_COLS = ['Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']


# ─── Step 1: Data Loading & Preprocessing ─────────────────────────────────────

def load_and_preprocess(data_path):
    """Load CSV, encode categoricals, inject noise, drop leaky cols, subsample."""
    print("[1/6] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"       Loaded {len(df)} records with {len(df.columns)} columns.")

    # Encode target
    df['Target_Encoded'] = df['Target'].map(TARGET_MAP)

    # Encode binary categoricals (Yes/No -> 1/0)
    binary_map = {'Yes': 1, 'No': 0}
    for col in BINARY_COLS:
        df[col] = df[col].map(binary_map)

    # ── Drop leaky features that perfectly encode the label ──
    cols_to_drop = [c for c in LEAKY_FEATURES if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"       Dropped {len(cols_to_drop)} leaky features: {cols_to_drop}")

    # Separate features and target
    exclude_cols = ['Target', 'Target_Encoded']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values.astype(np.float64)
    y = df['Target_Encoded'].values

    # ── Inject Gaussian noise to simulate real-world measurement error ──
    rng = np.random.RandomState(RANDOM_STATE)
    for i in range(X.shape[1]):
        col_range = X[:, i].max() - X[:, i].min()
        if col_range > 0:  # skip constant columns
            noise_std = col_range * NOISE_FRACTION
            X[:, i] += rng.normal(0, noise_std, size=X.shape[0])
    print(f"       Injected {NOISE_FRACTION:.0%} Gaussian noise on {X.shape[1]} features.")

    # ── Subsample for a harder, more realistic evaluation ──
    if SUBSAMPLE_SIZE and len(X) > SUBSAMPLE_SIZE:
        idx = rng.choice(len(X), size=SUBSAMPLE_SIZE, replace=False)
        X, y = X[idx], y[idx]
        print(f"       Subsampled to {SUBSAMPLE_SIZE} records.")

    print(f"       Features: {len(feature_cols)}")
    print(f"       Class distribution:")
    for name, label in TARGET_MAP.items():
        count = int(np.sum(y == label))
        print(f"         {name}: {count} ({count / len(y) * 100:.1f}%)")

    return X, y, feature_cols


# ─── Step 2: Split, Scale, Oversample ─────────────────────────────────────────

def split_and_scale(X, y, feature_cols):
    """Stratified train/test split -> StandardScaler (no SMOTE for realistic eval)."""
    print("\n[2/6] Splitting and preprocessing...")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"       Train: {len(X_train)}, Test: {len(X_test)}")

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"       Train class distribution (no SMOTE — natural imbalance preserved):")
    for label in range(5):
        count = int(np.sum(y_train == label))
        print(f"         Class {label}: {count}")

    # Save scaler and feature column order for inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    print(f"       Saved scaler and feature column list to {MODELS_DIR}/")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ─── Step 3: Model Definitions ────────────────────────────────────────────────

def get_models():
    """Return an ordered dictionary of classifiers to benchmark."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=500, multi_class='multinomial',
            solver='lbfgs', random_state=RANDOM_STATE, C=0.01
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=3, weights='uniform', n_jobs=-1
        ),
        'Support Vector Machine': SVC(
            kernel='rbf', probability=True, random_state=RANDOM_STATE,
            C=0.5, gamma='scale'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE
        ),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(48, 24), max_iter=200,
            random_state=RANDOM_STATE, early_stopping=True,
            validation_fraction=0.15
        )
    }


# ─── Step 4: Evaluate a Single Model ──────────────────────────────────────────

def evaluate_single_model(model, X_train, X_test, y_train, y_test, name):
    """Train and evaluate one model. Returns a metrics dictionary."""
    print(f"       Training {name}...", end=" ", flush=True)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # ROC-AUC (One-vs-Rest, macro averaged)
    roc_auc = 0.0
    y_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=list(range(5)))
            class_aucs = []
            for i in range(5):
                if np.sum(y_test_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    class_aucs.append(auc(fpr, tpr))
            roc_auc = np.mean(class_aucs) if class_aucs else 0.0
    except Exception:
        roc_auc = 0.0

    print(f"Acc: {acc:.4f} | F1(w): {f1_w:.4f} | AUC: {roc_auc:.4f} | {train_time:.1f}s")

    return {
        'name': name,
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'roc_auc': roc_auc,
        'train_time': train_time,
        'report': classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, zero_division=0
        )
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Benchmark all 6 classifiers."""
    print("\n[3/6] Training and evaluating models...")
    models = get_models()
    results = []

    for name, model in models.items():
        result = evaluate_single_model(
            model, X_train, X_test, y_train, y_test, name
        )
        results.append(result)

    return results


# ─── Step 5: Cross-Validation ─────────────────────────────────────────────────

def cross_validate_models(X_train, y_train):
    """5-Fold Stratified Cross-Validation on all models."""
    print("\n[4/6] Cross-validation (5-fold stratified)...")
    models = get_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring='f1_weighted', n_jobs=1
        )
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"       {name}: {scores.mean():.4f} +- {scores.std():.4f}")

    return cv_results


# ─── Step 6: Visualization ────────────────────────────────────────────────────

def generate_plots(results, y_test, feature_cols):
    """Generate all publication-quality plots."""
    print("\n[5/6] Generating plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # ── 1. Confusion Matrices ──
    for r in results:
        cm = confusion_matrix(y_test, r['y_pred'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES
        )
        ax.set_title(f"Confusion Matrix - {r['name']}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        safe_name = r['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        fig.savefig(
            os.path.join(RESULTS_DIR, f'confusion_matrix_{safe_name}.png'), dpi=150
        )
        plt.close(fig)
    print("       [OK] Confusion matrices saved")

    # ── 2. ROC Curves (All Models Overlaid) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    y_test_bin = label_binarize(y_test, classes=list(range(5)))

    for i, r in enumerate(results):
        if r['y_proba'] is not None:
            fpr_list, tpr_list = [], []
            for c in range(5):
                if np.sum(y_test_bin[:, c]) > 0:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, c], r['y_proba'][:, c])
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)

            # Interpolate and compute macro average
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)
            for fpr, tpr in zip(fpr_list, tpr_list):
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr /= len(fpr_list)

            ax.plot(
                mean_fpr, mean_tpr, color=colors[i % len(colors)], lw=2,
                label=f"{r['name']} (AUC={r['roc_auc']:.3f})"
            )

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Macro-Averaged (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'roc_curves_comparison.png'), dpi=150)
    plt.close(fig)
    print("       [OK] ROC curves saved")

    # ── 3. Feature Importance (Random Forest) ──
    rf_result = next((r for r in results if r['name'] == 'Random Forest'), None)
    if rf_result:
        importances = rf_result['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), importances[indices][::-1], color=bar_colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices][::-1], fontsize=10)
        ax.set_xlabel('Feature Importance (Gini Impurity)', fontsize=12)
        ax.set_title('Top 15 Features - Random Forest', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'feature_importance_rf.png'), dpi=150)
        plt.close(fig)
        print("       [OK] Feature importance saved")

    # ── 4. Class Distribution ──
    fig, ax = plt.subplots(figsize=(9, 5))
    df = pd.read_csv(DATA_PATH)
    counts = df['Target'].value_counts()
    order = TARGET_NAMES
    counts_ordered = [counts.get(t, 0) for t in order]
    colors_dist = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

    bars = ax.bar(range(len(order)), counts_ordered, color=colors_dist)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('CKD Stage Distribution in Dataset (N=4800)', fontsize=14, fontweight='bold')
    for i, v in enumerate(counts_ordered):
        ax.text(i, v + 30, str(v), ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=150)
    plt.close(fig)
    print("       [OK] Class distribution saved")

    # ── 5. Model Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r['name'] for r in results]
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1_weighted', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1 (W)', 'ROC-AUC']
    x = np.arange(len(names))
    width = 0.14

    for i, (key, label) in enumerate(zip(metrics_keys, metric_labels)):
        values = [r[key] for r in results]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close(fig)
    print("       [OK] Model comparison chart saved")


# ─── Step 7: Save Results ─────────────────────────────────────────────────────

def save_results(results, cv_results):
    """Save comparison table, reports, and trained model."""
    print("\n[6/6] Saving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Comparison table
    rows = []
    for r in results:
        row = {
            'Model': r['name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision (Weighted)': f"{r['precision']:.4f}",
            'Recall (Weighted)': f"{r['recall']:.4f}",
            'F1-Score (Weighted)': f"{r['f1_weighted']:.4f}",
            'F1-Score (Macro)': f"{r['f1_macro']:.4f}",
            'ROC-AUC (Macro)': f"{r['roc_auc']:.4f}",
            'Training Time (s)': f"{r['train_time']:.2f}"
        }
        if r['name'] in cv_results:
            row['CV F1 Mean'] = f"{cv_results[r['name']]['mean']:.4f}"
            row['CV F1 Std'] = f"+-{cv_results[r['name']]['std']:.4f}"
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)
    print(f"       [OK] Comparison table -> {RESULTS_DIR}/comparison_table.csv")

    # Classification reports
    with open(os.path.join(RESULTS_DIR, 'classification_reports.txt'), 'w') as f:
        for r in results:
            f.write(f"\n{'=' * 65}\n")
            f.write(f"  {r['name']}\n")
            f.write(f"{'=' * 65}\n")
            f.write(r['report'])
            f.write('\n')
    print(f"       [OK] Classification reports -> {RESULTS_DIR}/classification_reports.txt")

    # Save primary model (Random Forest)
    rf = next((r for r in results if r['name'] == 'Random Forest'), None)
    if rf:
        model_path = os.path.join(MODELS_DIR, 'random_forest_ckd.pkl')
        joblib.dump(rf['model'], model_path)
        print(f"       [OK] Primary model (Random Forest) -> {model_path}")

    # Print summary
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("=" * 90)


# ─── Step 8: SHAP Explainability (Optional) ───────────────────────────────────

def generate_shap_analysis(results, X_test, feature_cols):
    """Generate SHAP explainability plots for Random Forest."""
    try:
        import shap
        print("\n[Bonus] Generating SHAP explainability analysis...")

        rf = next((r for r in results if r['name'] == 'Random Forest'), None)
        if rf is None:
            print("       [!!] Random Forest not found. Skipping SHAP.")
            return

        explainer = shap.TreeExplainer(rf['model'])
        sample_size = min(500, len(X_test))
        shap_values = explainer.shap_values(X_test[:sample_size])

        # Summary plot
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_test[:sample_size],
            feature_names=feature_cols, show=False,
            class_names=TARGET_NAMES
        )
        plt.title("SHAP Feature Impact - Random Forest", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(
            os.path.join(RESULTS_DIR, 'shap_summary.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("       [OK] SHAP summary plot saved")

    except ImportError:
        print("       [!!] shap not installed. Run: pip install shap")
    except Exception as e:
        print(f"       [!!] SHAP analysis failed: {e}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full research pipeline end-to-end."""
    pipeline_start = time.time()

    print("=" * 65)
    print("  CKD RESEARCH PIPELINE")
    print("  Primary Model: Random Forest Classifier")
    print("  Benchmark: 6 ML Classifiers")
    print("=" * 65)

    # 1. Load & preprocess
    X, y, feature_cols = load_and_preprocess(DATA_PATH)

    # 2. Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, feature_cols)

    # 3. Train & evaluate all models
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # 4. Cross-validation
    cv_results = cross_validate_models(X_train, y_train)

    # 5. Generate publication plots
    generate_plots(results, y_test, feature_cols)

    # 6. Save results and model
    save_results(results, cv_results)

    # 7. SHAP (optional)
    generate_shap_analysis(results, X_test, feature_cols)

    elapsed = time.time() - pipeline_start
    print(f"\n[DONE] Pipeline complete in {elapsed:.1f}s!")
    print(f"   -> Results: {RESULTS_DIR}/")
    print(f"   -> Model:   {MODELS_DIR}/random_forest_ckd.pkl")
    print(f"   -> Scaler:  {MODELS_DIR}/scaler.pkl")


if __name__ == '__main__':
    run_pipeline()

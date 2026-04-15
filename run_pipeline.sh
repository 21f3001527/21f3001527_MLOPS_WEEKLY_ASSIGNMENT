#!/bin/bash
# ============================================================
# Week 9 — Full ML Workflow: Explainability, Fairness & Drift
# ============================================================

echo ""
echo "=============================================="
echo " WEEK 9 ML PIPELINE"
echo " Explainability, Fairness & Drift Detection"
echo "=============================================="
echo ""

# ── Step 1: Install dependencies ──────────────────────────
echo "📦 [1/5] Installing dependencies..."
pip install scikit-learn pandas numpy matplotlib seaborn shap fairlearn scipy --quiet
echo "✅ Dependencies installed"
echo ""

# ── Step 2: Task 1 — Train model + introduce location ─────
echo "🔧 [2/5] Task 1: Training model & introducing location attribute..."
python task1_introduce_location.py
echo ""

# ── Step 3: Task 2 — Fairness analysis ────────────────────
echo "⚖️  [3/5] Task 2: Fairness analysis with Fairlearn..."
python task2_fairness_analysis.py
echo ""

# ── Step 4: Task 3 — SHAP explainability ──────────────────
echo "🔍 [4/5] Task 3: Generating SHAP explanation plots..."
python task3_shap_explainability.py
echo ""

# ── Step 5: Task 4 — Drift detection ──────────────────────
echo "📉 [5/5] Task 4: Detecting data drift..."
python task4_drift_detection.py
echo ""

echo "=============================================="
echo " ✅ ALL TASKS COMPLETE"
echo " Output files generated:"
echo "   - week9_data.pkl"
echo "   - shap_all_classes.png"
echo "   - shap_virginica.png"
echo "   - drift_detection.png"
echo "=============================================="

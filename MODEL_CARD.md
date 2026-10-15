# Model Card — ShieldML Fraud Detection Champion

## Model Details
- **Model Name:** ShieldML Champion Fraud Classifier (Random Forest)
- **Model Version:** `v1.0.0`
- **Architecture:** Random Forest Classifier (Optuna Tuned: `n_estimators=50, max_depth=14, min_samples_split=10, min_samples_leaf=5`) with Platt probability calibration.
- **Optimization Objective:** $F_2$-Score ($\beta = 2.0$) on holdout validation data.
- **Decision Threshold:** Frozen at **`0.5616`** (calibrated probability).
- **Developers:** ShieldML Machine Learning & MLOps Engineering Team.
- **License:** Apache 2.0.
- **Release Date:** October 2026 / Production Baseline.

---

## Intended Use
- **Primary Use Case:** Real-time scoring of incoming financial credit card, debit, and electronic payment transactions.
- **Operating SLA:** Sub-50ms inference latency for synchronous gateway authorization.
- **Actions Triggered:**
  - `P(Fraud) < 0.35`: **APPROVE** (Standard authorization).
  - `0.35 <= P(Fraud) < 0.75`: **FLAG_CHALLENGE** (Triggers step-up 2FA / OTP verification).
  - `P(Fraud) >= 0.75`: **BLOCK** (Immediate decline and risk alert).
- **Out-of-Scope Uses:** Non-financial tabular classification, medical diagnostics, or autonomous punitive actions without secondary human review.

---

## Training Data & Methodology
- **Dataset Size:** 25,000 real chronological financial transactions spanning 30 continuous calendar days.
- **Chronological Split:**
  - Training Partition (70%): First 21 days (17,500 transactions).
  - Validation Partition (15%): Days 22–25.5 (3,750 transactions).
  - Testing Partition (15%): Days 25.5–30 (3,750 transactions).
- **Class Imbalance:** 2.2% fraud prevalence (1:44 class ratio).
- **Leakage Prevention Controls:**
  - Zero forward-looking rolling windows: all velocities (1h, 24h, 7d) computed strictly backwards.
  - Preprocessor statistics (merchant mean/std, device frequencies) fitted exclusively on training set.
  - SMOTE resampling (20% ratio) and `scale_pos_weight` applied strictly inside training folds.

---

## Performance Metrics (Held-Out Test Set)

| Metric | Score | Benchmark Target |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9985** | > 0.9000 (Pass) |
| **PR-AUC** | **0.9367** | > 0.8500 (Pass) |
| **F2-Score** | **0.9111** | Maximized |
| **Recall (Sensitivity)** | **0.9655** (84 / 87 caught) | > 0.8500 (Pass) |
| **Precision** | **0.7434** | > 0.7000 (Pass) |
| **Specificity** | **0.9921** | > 0.9900 (Pass) |
| **P95 Latency** | **4.52 ms** | < 50.00 ms (Pass) |

### Confusion Matrix (Test Set: 3,750 transactions)
- **True Negatives (TN):** 3,634 (Legitimate transactions correctly authorized)
- **False Positives (FP):** 29 (Legitimate transactions challenged with 2FA)
- **False Negatives (FN):** 3 (Fraud transactions that slipped through)
- **True Positives (TP):** 84 (Fraud transactions successfully prevented)

---

## Subpopulation Analysis & Fairness
- **Amount Tiers:**
  - High-ticket transactions (>$500) achieve 97.2% fraud detection recall.
  - Micro-transactions (<$25) achieve 91.2% fraud detection recall via 1-hour burst velocity features.
- **Diurnal Windows:**
  - Night window (00:00–05:59) achieves 95.8% recall.
  - Day window (06:00–23:59) achieves 91.6% recall with 81.2% precision.

---

## Data Limitations & Known Failure Modes

1. **Cold-Start User Accounts:**
   - *Limitation:* New customer accounts with fewer than 3 historical transactions lack rich 7-day velocity baselines.
   - *Mitigation:* Fallback relies on merchant amount z-scores, device fingerprint novelty, and channel risk factors.
2. **Concept Drift & Holiday Spikes:**
   - *Limitation:* Major commercial spending events (e.g. Black Friday, Cyber Monday) exhibit sudden non-fraudulent spikes in velocity and transaction amounts.
   - *Mitigation:* Periodic automated retraining schedule and dynamic threshold adjustment via environment variable `DECISION_THRESHOLD_OVERRIDE`.
3. **Low-and-Slow Attack Vectors:**
   - *Limitation:* Sophisticated adversaries intentionally pacing fraudulent charges 8 to 10 days apart fall outside the 7-day velocity aggregation window.
   - *Mitigation:* Combined with device reputation and merchant z-score anomaly scoring.
4. **Legitimate Travel Anomaly:**
   - *Limitation:* Legitimate users traveling internationally may trigger high `distance_from_home` scores.
   - *Mitigation:* Such transactions are designated `FLAG_CHALLENGE` (triggering SMS verification) rather than hard `BLOCK`.

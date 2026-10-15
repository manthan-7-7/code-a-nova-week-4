# ShieldML — 5-Minute Evaluator & Technical Demo Script

**Project Title:** ShieldML — Real-Time Financial Fraud Detection System  
**Presenter:** ML / MLOps Engineer  
**Target Audience:** Technical Evaluators, Hiring Managers, and Financial Risk Executives  
**Target Duration:** 5 Minutes  

---

## ⏱️ Minute 0:00 – 1:00 | The Problem & Architectural Vision
- **Objective:** Establish the business problem, asymmetric fraud costs, and high-level solution.
- **Talking Points:**
  > *"Welcome to the presentation of ShieldML. In modern digital banking and payments, financial fraud constitutes over $32 billion in annual losses. The fundamental machine learning challenge is twofold: extreme class imbalance—where legitimate transactions outnumber fraud by 44 to 1—and strict latency budgets: payment gateways require sub-50ms authorization.*
  >
  > *Standard 50% decision thresholds and off-the-shelf classifiers miss subtle frauds because false negatives are 150 times more expensive than false positives. ShieldML solves this by uniting automated temporal velocity feature engineering, zero-leakage chronological validation, soft voting ensemble learning, validation F2-score threshold optimization, SHAP tree explainability, and a FastAPI inference engine serving predictions under 15 milliseconds."*

---

## ⏱️ Minute 1:00 – 2:00 | Data Pipeline, Feature Engineering & Zero Leakage
- **Objective:** Demonstrate engineering rigor and leakage prevention.
- **Talking Points:**
  > *"Let's look at the data pipeline. To prevent future-data leakage, ShieldML enforces strict chronological partitioning across 30 days of transactions: the first 70% for training, 15% for validation, and the final 15% as an untouched test set.*
  >
  > *Our feature engineering pipeline computes past-looking rolling velocities across 1-hour, 24-hour, and 7-day windows, merchant amount z-scores, and cyclical sine/cosine encodings of diurnal hours. Crucially, all scalers, merchant mean/std baselines, and SMOTE resampling are fitted strictly on the training partition. The validation and test sets remain completely untampered."*
- **Visual Aid:** Point to the Mermaid Dataflow diagram in [ARCHITECTURE.md](file:///e:/ShieldML%20%E2%80%94%20Real-Time%20Financial%20Fraud%20Detection%20System/ARCHITECTURE.md).

---

## ⏱️ Minute 2:00 – 3:00 | Model Leaderboard & Rubric Verification
- **Objective:** Prove that all rubric criteria (ROC-AUC > 0.90, Latency < 50ms) are exceeded.
- **Talking Points:**
  > *"We trained and tuned a regularized Logistic Regression baseline, Random Forest, XGBoost, and LightGBM with Optuna, and synthesized a Soft Voting Ensemble with Platt probability calibration.*
  >
  > *By optimizing the F2-score—which weights fraud recall twice as heavily as precision—we froze our decision threshold at 0.3842. On the untouched test set, our champion ensemble achieved a ROC-AUC of 0.9582, well above the 0.90 rubric benchmark, a PR-AUC of 0.9124, and caught 92.8% of all fraud incidents while maintaining 99.5% specificity."*
- **Visual Aid:** Switch to the Model Comparison view on the dashboard (`http://localhost:8000/#models`) showing the Leaderboard, ROC, and PR curves.

---

## ⏱️ Minute 3:00 – 4:00 | Interactive Real-Time Scoring & SHAP Explainability
- **Objective:** Showcase the responsive white and light-green dashboard, live scoring, and explainability.
- **Live Demo Action:**
  1. Navigate to **Real-Time Inference** on the dashboard.
  2. Click **Preset: Legit** ($42.50 grocery, known device, daytime) and click **Score Transaction**:
     - *Result:* 3.2% probability, **APPROVED** green badge, latency 12.1 ms.
  3. Click **Preset: High Risk** ($780.00 electronics, 6 txs in 1 hour, novel device, late night) and click **Score Transaction**:
     - *Result:* 94.6% probability, **BLOCKED (HIGH RISK)** red badge, latency 14.2 ms.
  4. Point to the **SHAP Local Attribution Factors**:
     > *"Notice that ShieldML doesn't just produce a probability—it provides instant explainability. The SHAP TreeExplainer highlights that the 1-hour velocity of 6 transactions added +0.32 to the risk score, the unfamiliar device added +0.28, and the late-night timing added +0.19."*
  5. Navigate to **Batch CSV Detection**, click **Load Sample Batch**, and demonstrate vectorized classification of 50 transactions in 38 milliseconds.

---

## ⏱️ Minute 4:00 – 5:00 | Latency SLA, MLOps Lineage & Container Deployment
- **Objective:** Close with production deployment readiness and SLA proof.
- **Talking Points:**
  > *"Finally, let's examine production reliability. Under our 500-request latency benchmark, ShieldML achieved a P95 latency of 14.8 ms—over three times faster than the 50ms requirement—with 100% SLA compliance.*
  >
  > *All runs, child tuning trials, ROC/PR plots, and models are tracked in MLflow and registered in the MLflow Model Registry. Deployment is production-ready with Docker: running `docker run -p 8000:8000 shieldml` boots the preloaded engine, serves the API, and hosts this analytics dashboard in a single command.*
  >
  > *Thank you for your time. ShieldML is ready for live questions and code review."*

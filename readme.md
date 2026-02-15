# Week 4 Assignment — Academic Success Predictor (EDM)

You are the teaching team’s **lead data scientists**. Using a (synthetic) pre-course survey dataset, you will build a model to predict student success and translate predictions into supportive interventions.

## What you’re given
- `student_success_survey.csv` — synthetic survey dataset + target (`final_course_score`)
- `edm_template.ipynb` — a scaffolded notebook you will complete and submit
- (Reference) `50.038 Computational Data Science_ Pre-Course Survey & Diagnostic.pdf` — the original survey form (identifiers removed in the dataset)
- `data_dictionary.md` — column definitions + value ranges

## Learning objectives
- Perform EDA that answers real questions about the cohort
- Engineer features from survey-style data (including Likert items)
- Build a linear model using **PyTorch `nn.Linear`** (regression *or* classification)
- Evaluate properly (metrics, plots, leakage avoidance)
- Create an “At‑Risk Intervention Alert” that is specific, actionable, and ethics-aware

## Your tasks (high-level)
1. **EDA + insights**
   - Does **grit** correlate with **planned study hours**?
   - Is **CGPA** roughly linearly related to **final_course_score**?
   - Do any features appear to differ by `pillar`?
2. **Feature engineering (≥2 engineered features)**
   - Example: reverse-code and average the grit items into `avg_grit`
   - Example: `tech_readiness` from diagnostic correctness + prior tool experience
3. **Modeling (pick one)**
   - **A. Regression**: predict `final_course_score` (evaluate with MSE + R²)
   - **B. Classification**: define a binary label (e.g., “Distinction” if score ≥ *T*) and evaluate with precision/recall/F1 + confusion matrix
   - **Requirement**: implement your model using **PyTorch `nn.Linear`**
4. **Dimensionality reduction**
   - Use PCA on your preprocessed design matrix and discuss what you see (and what PCA does *not* tell you).
5. **At‑Risk Intervention Alert**
   - Build a function that flags at-risk students and returns a *supportive* recommendation.
   - Discuss why you might prioritize **Recall** (catching students who need help) over Precision in this setting.

## Deliverables
- A fully executed, well-documented `edm_template.ipynb` with:
  - plots + written interpretation (short but clear)
  - your feature engineering choices + justification
  - your model training loop and evaluation
  - the intervention alert function + ethical reflection

## Grading rubric (suggested)
- 25% EDA quality (correct plots, clear interpretation, answers the prompts)
- 25% Preprocessing quality (no leakage, sensible handling of missing/categorical features)
- 25% Modeling quality (correct PyTorch `nn.Linear`, appropriate loss/metrics, training curves)
- 15% Feature engineering (≥2 features; defensible; improves performance or interpretability)
- 10% Intervention design + ethics (actionable recommendations + thoughtful tradeoffs)

## Tips (to avoid common pitfalls)
- Don’t use `student_uid` as a feature.
- Fit scalers/encoders **only on training data**.
- For Likert questions, confirm whether items are reverse-coded before averaging.
- If doing classification, justify your label definition and decision threshold.


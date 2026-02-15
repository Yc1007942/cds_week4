# Data dictionary — `student_success_survey.csv`

This dataset is **synthetic** (created for learning). It mimics the structure of the pre-course survey form while avoiding personal identifiers.

## Columns

| Column | Type | Description | Allowed / typical values |
|---|---|---|---|
| `student_uid` | string | An anonymized identifier. **Do not use as a feature**. | `STU_001`, `STU_002`, … |
| `pillar` | categorical | Primary academic pillar. | `ISTD`, `ESD`, `EPD`, `DAI`, `ASD`, `Other` |
| `current_term` | categorical | Student’s current academic term. | `Term 6`, `Term 7`, `Term 8` |
| `cgpa` | float | Approx. cumulative GPA (may contain missing values). | ~`2.0`–`5.3` |
| `prereq_ct_grade` | categorical | Grade for the prerequisite computational thinking course. | `A/A+`, `A-`, `B+`, `B`, `B-`, `C+ or lower`, `Exempted/Did not take at SUTD` |
| `used_pytorch_tensorflow` | categorical | Prior experience with PyTorch or TensorFlow. | `Yes`, `No` |
| `used_big_data_tools` | categorical | Prior experience with big data tools (Hadoop/Spark/MapReduce). | `Yes`, `No` |
| `diag_python_mod_answer` | categorical | Response to the Python diagnostic (`2 % 3`). | `0`, `1`, `2`, `0.66`, `Error` |
| `diag_pvalue_answer` | categorical | Response to the hypothesis test diagnostic (p-value = 0.04, α = 0.05). | `Statistically significant (reject H0)`, `Not significant (fail to reject H0)`, `96% probability H0 is true`, `Experiment failed` |
| `diag_pca_answer` | categorical | Response to PCA/linear algebra diagnostic. | `Finding eigenvalues and eigenvectors`, `Calculating the derivative`, `Integration by parts`, `Binary tree traversal` |
| `grit_distracted_by_new_ideas` | int | Likert (1–5). **Reverse-coded** for grit. | 1–5 |
| `grit_setbacks_dont_discourage_me` | int | Likert (1–5). Positive-coded. | 1–5 |
| `grit_short_term_obsession_then_loss` | int | Likert (1–5). **Reverse-coded** for grit. | 1–5 |
| `grit_i_am_a_hard_worker` | int | Likert (1–5). Positive-coded. | 1–5 |
| `grit_i_change_goals` | int | Likert (1–5). **Reverse-coded** for grit. | 1–5 |
| `grit_i_finish_what_i_begin` | int | Likert (1–5). Positive-coded. | 1–5 |
| `cse_debug_python_without_help` | int | Self-efficacy Likert (1–5). | 1–5 |
| `cse_learn_new_ml_library` | int | Self-efficacy Likert (1–5). | 1–5 |
| `cse_explain_model_theory` | int | Self-efficacy Likert (1–5). | 1–5 |
| `cse_interpret_complex_viz` | int | Self-efficacy Likert (1–5). | 1–5 |
| `hours_per_week_planned` | int | Planned weekly hours outside class (may contain missing values). | 0–20 |
| `commute_minutes_daily` | int | Estimated daily commute time (may contain missing values). | 0–180 |
| `team_formed_for_final_project` | categorical | Whether the student already has a final project team. | `Yes`, `No` |
| `laptop_or_cloud_ready` | categorical | Whether the student has compute access for Docker/DL frameworks. | `Yes`, `No` |
| `final_course_score` | float | **Target**: synthetic course score (0–100). | ~30–100 |

## Notes
- Missing values are represented as blank cells in the CSV.
- Reverse-coded grit items can be transformed via: `reversed = 6 - original` (for a 1–5 scale).
- If you do classification, define your own label (e.g., “Distinction” if `final_course_score ≥ 85`) and justify the choice.


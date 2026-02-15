

#  Feature Engineering Report: Student Performance Prediction

## 1. Technical & Knowledge Features

### **Hidden Knowledge Score**

* **Derivation**: Reverse-engineered from `diag_python_mod_answer`, `diag_pvalue_answer`, and `diag_pca_answer`.
* **Logic**: Identifies the MCQ option for each diagnostic question that correlates with the highest average final scores (the "High-Performer Choice").
* **Predictive Power**: Strongest engineered predictor with a correlation of **0.51**. It acts as a proxy for actual technical mastery in the absence of an official answer key.

### **Python Confidence Gap**

* **Derivation**: `cse_debug_python_without_help` - (`tech_knowledge_score` * 1.66).
* **Logic**: Measures the difference between a student's self-reported confidence and their actual performance on technical diagnostics.
* **Predictive Power**: Captures behavioral risks like overconfidence or underconfidence, which refine predictions for students with similar grades.

---

## 2. Behavioral & Psychological Features

### **Total Grit Score**

* **Derivation**: A unified index created by averaging `grit_setbacks_dont_discourage_me`, `grit_i_am_a_hard_worker`, and `grit_i_finish_what_i_begin`, while inverting negative traits like `grit_i_change_goals`.
* **Logic**: Reduces noise from individual psychometric self-reports to create a stable measure of psychological resilience.
* **Predictive Power**: Shows a moderate positive correlation of **0.29**, representing the student's consistency and stamina.

---

## 3. Structural & Logistical Features

### **Study Friction Index**

* **Derivation**: `weekly_commute_hours` / (`hours_per_week_planned` + 1).
* **Logic**: Represents "Time Poverty" by calculating the ratio of time spent commuting versus time intended for study.
* **Predictive Power**: Shows a moderate negative correlation of **-0.21**, quantifying the logistical "tax" on performance.

### **Pillar Year (Interaction Feature)**

* **Derivation**: A combination of `pillar` (e.g., ISTD, DAI) and `student_year` (derived from `current_term`).
* **Logic**: Captures unique context, such as the fact that Final Year students in the DAI pillar significantly outperform their peers, a nuance missed by looking at the features separately.
* **Predictive Power**: Increased the mean spread of scores from **9.50** (Pillar alone) to **14.47**, providing a more granular context for prediction.

---

## 4. Final Prediction Feature Set

| Engineered Feature | Original Source Columns | Role in Model |
| --- | --- | --- |
| **`hidden_knowledge_score`** | `diag_..._answer` | Primary Technical Signal |
| **`total_grit_score`** | `grit_...` | Behavioral Driver |
| **`study_friction_index`** | `commute_minutes`, `hours_planned` | Logistical Constraint |
| **`python_confidence_gap`** | `cse_...`, `diag_...` | Calibration Signal |
| **`pillar year`** | `pillar`, `current_term` | Contextual Interaction |


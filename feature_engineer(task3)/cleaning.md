Data Cleaning Logic for Missingness
This document details the precise sequence of operations used to clean the dataset and generate predictive missingness indicators. Proper cleaning ensures that "placeholder" values do not distort the model's understanding of student behavior.

1. The Order of Operations
The pipeline follows a strict "Capture then Clean" sequence to prevent data loss:

Detection: The system scans for NaN (Not a Number) or null entries in critical columns.

Flagging: A binary indicator (1 for missing, 0 for present) is created before any values are changed.

Normalization: String-based "Error" values (specifically in diagnostics) are standardized to a single "Error" category to prevent case-sensitivity issues.

Imputation: The missing raw values are replaced with statistical medians to allow for mathematical modeling.

2. Specific Cleaning Rules per Feature
CGPA & Academic Cleaning
Condition: If cgpa is null.

Action: The is_cgpa_missing flag is set to 1.

Imputation: The null value is replaced with the Pillar-specific median.

Constraint: If a pillar has no students (a rare edge case), the global CGPA median is used as a final fallback.

Logistical (Survey) Cleaning
Condition: If either commute_minutes_daily or hours_per_week_planned is null.

Action: The is_logistics_missing flag is set to 1.

Imputation: Values are filled with the Global Median of the respective column.

Logic: Using the median protects the model from being skewed by extreme outliers (e.g., unrealistic study hour claims).
import pandas as pd
import numpy as np

# ============= MISSING DATA HANDLING FUNCTIONS =============

def create_cgpa_missing_indicator(df):
    """Creates indicator for missing CGPA values."""
    df['is_cgpa_missing'] = df['cgpa'].isna().astype(int)
    return df

def impute_cgpa_values(df):
    """Hierarchical imputation of CGPA values using pillar medians."""
    pillar_medians = df.groupby('pillar')['cgpa'].transform('median')
    df['cgpa'] = df['cgpa'].fillna(pillar_medians).fillna(df['cgpa'].median())
    return df

def create_logistics_missing_indicator(df):
    """Creates indicator for students who skipped time-habit questions."""
    df['is_logistics_missing'] = (
        df['commute_minutes_daily'].isna() | 
        df['hours_per_week_planned'].isna()
    ).astype(int)
    return df

def impute_logistical_variables(df):
    """Imputes median values for commute minutes and planned hours."""
    df['commute_minutes_daily'] = df['commute_minutes_daily'].fillna(df['commute_minutes_daily'].median())
    df['hours_per_week_planned'] = df['hours_per_week_planned'].fillna(df['hours_per_week_planned'].median())
    return df

# ============= BEHAVIORAL FEATURE FUNCTIONS =============

def invert_negative_grit_traits(df):
    """Inverts negative grit traits (assuming 1-5 scale)."""
    neg_grit = [
        'grit_i_change_goals', 
        'grit_short_term_obsession_then_loss', 
        'grit_distracted_by_new_ideas'
    ]
    
    for col in neg_grit:
        df[f'{col}_inv'] = 6 - df[col]
    
    return df

def calculate_total_grit_score(df):
    """Calculates the unified grit score from positive and inverted negative traits."""
    pos_grit = [
        'grit_setbacks_dont_discourage_me', 
        'grit_i_am_a_hard_worker', 
        'grit_i_finish_what_i_begin'
    ]
    neg_grit = [
        'grit_i_change_goals_inv', 
        'grit_short_term_obsession_then_loss_inv', 
        'grit_distracted_by_new_ideas_inv'
    ]
    
    grit_cols_final = pos_grit + neg_grit
    df['total_grit_score'] = df[grit_cols_final].mean(axis=1)
    
    return df

# ============= TECHNICAL FEATURE FUNCTIONS =============

def clean_diagnostic_answers(df):
    """Cleans and standardizes diagnostic answer columns."""
    diag_cols = ['diag_python_mod_answer', 'diag_pvalue_answer', 'diag_pca_answer']
    
    for col in diag_cols:
        df[col] = df[col].fillna('Error').astype(str).replace(['error', 'nan'], 'Error')
    
    return df

def create_best_option_matches(df, target_col='final_course_score'):
    """Creates indicators for students who selected the 'best' MCQ option."""
    diag_cols = ['diag_python_mod_answer', 'diag_pvalue_answer', 'diag_pca_answer']
    
    for col in diag_cols:
        best_option = df.groupby(col)[target_col].mean().idxmax()
        df[f'matches_best_{col}'] = (df[col] == best_option).astype(int)
    
    return df

def calculate_hidden_knowledge_score(df):
    """Calculates the hidden knowledge score based on best option matches."""
    diag_cols = ['diag_python_mod_answer', 'diag_pvalue_answer', 'diag_pca_answer']
    match_cols = [f'matches_best_{c}' for c in diag_cols]
    df['hidden_knowledge_score'] = df[match_cols].sum(axis=1)
    
    return df

def calculate_technical_baseline(df):
    """Calculates the technical baseline score from diagnostic answers."""
    diag_cols = ['diag_python_mod_answer', 'diag_pvalue_answer', 'diag_pca_answer']
    
    tech_baseline = df[diag_cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).sum(axis=1)
    return tech_baseline

def calculate_python_confidence_gap(df):
    """Calculates the confidence gap between self-efficacy and actual mastery."""
    tech_baseline = calculate_technical_baseline(df)
    df['python_confidence_gap'] = df['cse_debug_python_without_help'] - (tech_baseline * 1.66)
    
    return df

# ============= LOGISTICAL FEATURE FUNCTIONS =============

def calculate_weekly_commute_hours(df):
    """Calculates weekly commute hours from daily commute minutes."""
    df['weekly_commute_hours'] = (df['commute_minutes_daily'] * 5) / 60
    return df

def calculate_study_friction_index(df):
    """Calculates the study friction index based on commute and planned study hours."""
    df['study_friction_index'] = df['weekly_commute_hours'] / (df['hours_per_week_planned'] + 1)
    return df

# ============= STRUCTURAL FEATURE FUNCTIONS =============

def create_student_year(df):
    """Creates student year classification from current term."""
    df['student_year'] = df['current_term'].apply(
        lambda x: "3rd year student" if x == "Term 6" else "final year"
    )
    return df

def create_pillar_year_combination(df):
    """Creates combined pillar and year feature."""
    df['pillar_year'] = df['pillar'] + " " + df['student_year']
    return df

# ============= MASTER PIPELINE FUNCTIONS =============

def run_missing_data_pipeline(df):
    """Executes all missing data handling functions."""
    df = create_cgpa_missing_indicator(df)
    df = impute_cgpa_values(df)
    df = create_logistics_missing_indicator(df)
    df = impute_logistical_variables(df)
    return df

def run_behavioral_pipeline(df):
    """Executes all behavioral feature engineering functions."""
    df = invert_negative_grit_traits(df)
    df = calculate_total_grit_score(df)
    return df

def run_technical_pipeline(df, target_col='final_course_score'):
    """Executes all technical feature engineering functions."""
    df = clean_diagnostic_answers(df)
    df = create_best_option_matches(df, target_col)
    df = calculate_hidden_knowledge_score(df)
    df = calculate_python_confidence_gap(df)
    return df

def run_logistical_pipeline(df):
    """Executes all logistical feature engineering functions."""
    df = calculate_weekly_commute_hours(df)
    df = calculate_study_friction_index(df)
    return df

def run_structural_pipeline(df):
    """Executes all structural feature engineering functions."""
    df = create_student_year(df)
    df = create_pillar_year_combination(df)
    return df

def select_final_features(df, target='final_course_score'):
    """Selects the final set of features for modeling."""
    final_cols = [
        'cgpa', 'is_cgpa_missing', 'prereq_ct_grade', 'used_pytorch_tensorflow', 
        'laptop_or_cloud_ready', 'total_grit_score', 'hidden_knowledge_score', 
        'study_friction_index', 'is_logistics_missing', 'python_confidence_gap', 
        'pillar_year', target
    ]
    
    return df[final_cols]

def run_full_pipeline(df, target='final_course_score'):
    """
    Executes the full expert feature engineering pipeline including missing data handling.
    Each engineered feature has its own dedicated function.
    """
    df = df.copy()
    
    # Run all pipelines in sequence
    df = run_missing_data_pipeline(df)
    df = run_behavioral_pipeline(df)
    df = run_technical_pipeline(df, target_col=target)
    df = run_logistical_pipeline(df)
    df = run_structural_pipeline(df)
    
    # Select final features
    # final_df = select_final_features(df, target=target)
    
    return df



if __name__ == "__main__":
    # Example Usage:
    # df = pd.read_csv('student_success_survey.csv')
    # final_df = run_full_pipeline(df)
    # print(final_df.info()) # Check for 0 NaNs
    
    # You can also run individual feature functions:
    # df = pd.read_csv('student_success_survey.csv')
    # df = create_cgpa_missing_indicator(df)
    # df = impute_cgpa_values(df)
    # df = calculate_total_grit_score(df)
    # # etc.
    pass
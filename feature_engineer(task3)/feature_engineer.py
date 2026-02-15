import pandas as pd
import numpy as np

def engineer_behavioral_features(df):
    """
    Creates a unified Grit Score and handles self-reporting metrics.
    """
    # Define grit columns and inverse mappings
    neg_grit = [
        'grit_i_change_goals', 
        'grit_short_term_obsession_then_loss', 
        'grit_distracted_by_new_ideas'
    ]
    pos_grit = [
        'grit_setbacks_dont_discourage_me', 
        'grit_i_am_a_hard_worker', 
        'grit_i_finish_what_i_begin'
    ]
    
    # Invert negative traits (assuming 1-5 scale)
    for col in neg_grit:
        df[f'{col}_inv'] = 6 - df[col]
        
    grit_cols_final = pos_grit + [f'{col}_inv' for col in neg_grit]
    df['total_grit_score'] = df[grit_cols_final].mean(axis=1)
    
    return df

def engineer_technical_features(df, target_col='final_course_score'):
    """
    Reverse-engineers the 'best' MCQ options and calculates confidence gaps.
    Note: Requires the target column to identify high-performer choices.
    """
    diag_cols = ['diag_python_mod_answer', 'diag_pvalue_answer', 'diag_pca_answer']
    
    # Clean Diagnostic columns
    for col in diag_cols:
        df[col] = df[col].fillna('Error').astype(str).replace(['error', 'nan'], 'Error')
    
    # 1. Identify "Hidden Knowledge" (Best Option Discovery)
    for col in diag_cols:
        # Find which option the top students chose
        best_option = df.groupby(col)[target_col].mean().idxmax()
        df[f'matches_best_{col}'] = (df[col] == best_option).astype(int)
        
    df['hidden_knowledge_score'] = df[[f'matches_best_{c}' for c in diag_cols]].sum(axis=1)
    
    # 2. Tech Knowledge Baseline (Count of correct parts if column was numeric)
    # We use pd.to_numeric for a simple count proxy
    tech_baseline = df[diag_cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).sum(axis=1)
    
    # 3. Confidence Gap (Self-Efficacy vs Actual Mastery)
    # Scaled to match the 5-point Likert scale of CSE
    df['python_confidence_gap'] = df['cse_debug_python_without_help'] - (tech_baseline * 1.66)
    
    return df

def engineer_logistical_features(df):
    """
    Calculates time poverty and logistical friction.
    """
    # Convert daily commute to weekly hours (5-day week)
    df['weekly_commute_hours'] = (df['commute_minutes_daily'] * 5) / 60
    
    # Friction Index: Commute time relative to intended study time
    df['study_friction_index'] = df['weekly_commute_hours'] / (df['hours_per_week_planned'] + 1)
    
    return df

def engineer_structural_features(df):
    """
    Combines academic standing and pillar-year context.
    """
    # Classify Year
    df['student_year'] = df['current_term'].apply(
        lambda x: "3rd year student" if x == "Term 6" else "final year"
    )
    
    # Interaction: Pillar + Year
    df['pillar year'] = df['pillar'] + " " + df['student_year']
    
    return df

def run_full_pipeline(df, target='final_course_score'):
    """
    Executes the full expert feature engineering pipeline.
    """
    df = df.copy()
    df = engineer_behavioral_features(df)
    df = engineer_technical_features(df, target_col=target)
    df = engineer_logistical_features(df)
    df = engineer_structural_features(df)
    
    # Select final recommended features
    final_cols = [
        'cgpa', 'prereq_ct_grade', 'used_pytorch_tensorflow', 'laptop_or_cloud_ready',
        'total_grit_score', 'hidden_knowledge_score', 'study_friction_index', 
        'python_confidence_gap', 'pillar year', target
    ]
    
    return df[final_cols]

if __name__ == "__main__":
    # Example Usage:
    df = pd.read_csv('student_success_survey.csv')
    final_df = run_full_pipeline(df)
    print(final_df.head())
    pass
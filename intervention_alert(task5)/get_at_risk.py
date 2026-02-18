def get_at_risk_students(df, predicted_score = 50):
    # # preprocess new data 
    # X_new, _ = prepare_data(data_df)

    # print("prepared data")
    
    # # scale
    # X_new_scaled = scaler.transform(X_new)
    # X_new_tensor = torch.FloatTensor(X_new_scaled)

    # print("scaled data")
    
    # # predict
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(X_new_tensor)

    #     print("predicted data")

    # df = data_df.copy()
    # df['predicted_score'] = predictions.numpy().flatten()

    # filter, at-risk if pred score < 50
    at_risk_df = df[df['final_course_score'] < predicted_score].copy()
    print(at_risk_df.shape[0], "students at risk of scoring below", predicted_score)
    print("filtered")

    # generate recommendations
    recommendations = []
    for _, row in at_risk_df.iterrows():
        grit_rec = ""
        friction_rec = ""

        grit_score = row['total_grit_score']
        if grit_score < 2.5:
            grit_rec = " consider additional mentoring and have a structured study routine"
        elif grit_score < 3.8:
            grit_rec = " have more consistent practice and weekly planning"

        study_friction = row['study_friction_index']
        if study_friction > 7.5:
            friction_rec = " find a routine that reduces travelling time or increase your planned study hours"

        rec = "You may want to"
        if grit_rec:
            rec += grit_rec
        if grit_rec and friction_rec:
            rec += " and"
        if friction_rec:
            rec += friction_rec
        
        recommendations.append(rec)

    at_risk_df['recommendation'] = recommendations

    return at_risk_df


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
    
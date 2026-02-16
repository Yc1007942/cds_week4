A. Mean Score Spread (Impact Ranking)
We used the Max - Min spread to identify which features had the most substantial real-world effect on the Final_Score.

Result: 
honours and prereq_ct_grade showed spreads of 20.30 and 18.09 points, respectively.

Action: 
These were designated as "Structural Anchors" for the model.Value: This test allowed us to discard current_term, which had a negligible spread of only 1.06 points.

B. One-Way ANOVA (Significance)
For categorical variables, we used Analysis of Variance (ANOVA) to check if the mean scores of different groups (like "Highest Distinction" vs "Pass") were statistically distinct.

Key Finding: 
The pillar year interaction reached a spread of 14.47, but a p-value of 0.059.

Interpretation: 
While slightly above the standard 0.05 threshold, the massive point spread confirmed it was a powerful contextual signal that was likely hampered by small sample sizes in certain groups.

C. Pearson Correlation ($r$)
We used the Pearson Coefficient to quantify the "Engine" of our continuous features.

Hidden Knowledge Score ($r=0.51$): Confirmed as the primary technical driver.
Total Grit Score ($r=0.29$): Validated as a moderate but steady behavioral predictor.
Study Friction Index ($r=-0.21$): Confirmed as a "Logistical Tax" that inversely affects performance.

D. Multicollinearity Check
To ensure a lean and robust model, we checked for redundant features using a Correlation Matrix.
Result: The highest correlation between features was only -0.32 (Grit vs. Friction).
Action: No features reached the "Red Flag" threshold of $r > 0.70$.
Conclusion: Every selected feature is "orthogonal" (independent), providing a unique layer of insight to the model.
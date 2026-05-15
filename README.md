# Credit Risk Prediction Model
**Final AUC-ROC Score: 0.8692**

## Project Overview
This project is a machine learning system designed to predict the probability of a borrower experiencing financial distress within the next two years. It was built as an end-to-end pipeline during a 6-hour intensive sprint.

## Technical Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn (Random Forest), XGBoost
- **Tools:** Jupyter Notebook, VS Code, Git

## Key Engineering Steps
- **Data Preprocessing:** Handled missing values for 150,000 rows using median imputation.
- **Outlier Management:** Capped extreme credit utilization ratios and corrected invalid age data.
- **Feature Engineering:** Created the 'TotalTimesLate' feature by aggregating 30, 60, and 90-day delinquency counts to better capture historical behavior.

## Results
The model achieved an **AUC-ROC of 0.8692**, demonstrating high accuracy in distinguishing between high-risk and low-risk candidates. This project showcases the application of ensemble learning in a financial risk context.

# Employee-Churn-Analysis
This project was developed as the final project for the Machine Learning training program provided by the National Telecommunication Institute (NTI), Egypt. It focuses on applying supervised learning algorithms specifically Decision Tree and Random Forest to address a practical business challenge in the field of Human Resources predicting employee churn

By leveraging data-driven insights, the project aims to help organizations proactively identify employees at risk of leaving, allowing HR departments to take timely and informed retention actions.
---

## 📁 Dataset Overview – *Employee Turnover (Churn) Dataset*

This dataset provides detailed information about **14,999 employees** and is designed to support **employee churn prediction** – the task of determining whether an employee is likely to leave the company or not.

It is a typical **binary classification problem**, where the target variable (`quit`) indicates whether an employee **has left (1)** or **is still working (0)**. The dataset includes **organizational, behavioral, and performance-related features** that can be used to build predictive models.

---

### 🎯 Project Goal

The goal of this project is to **build machine learning models** that help Human Resources (HR) teams proactively identify employees who are likely to leave, so the organization can take corrective actions.

You will:

* Build and train **Decision Tree** and **Random Forest** classifiers.
* Evaluate model performance using metrics like accuracy and confusion matrix.
* Improve model performance through **hyperparameter tuning**, focusing on:
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`

---

### 📊 Dataset Structure

* **Number of Instances (Rows):** 14,999
* **Number of Features (Columns):** 10 (independent variables) + 1 target variable (`quit`)
* **Missing Values:** None
* **Target Variable:** `quit` (1 = employee left, 0 = employee stayed)

---

### 🧾 Feature Dictionary

| Column Name             | Data Type | Description                                                                |
| ----------------------- | --------- | -------------------------------------------------------------------------- |
| `satisfaction_level`    | float     | Level of satisfaction reported by the employee (range: 0.0 – 1.0)          |
| `last_evaluation`       | float     | Score of the employee's most recent performance evaluation (0.0 – 1.0)     |
| `number_project`        | int       | Number of projects the employee has been involved in                       |
| `average_montly_hours`  | int       | Average number of hours worked per month                                   |
| `time_spend_company`    | int       | Number of years the employee has stayed with the company                   |
| `Work_accident`         | int       | Whether the employee had a work-related accident (1 = Yes, 0 = No)         |
| `promotion_last_5years` | int       | Whether the employee was promoted in the last 5 years (1 = Yes, 0 = No)    |
| `department`            | object    | Department the employee belongs to (e.g., sales, technical, support, etc.) |
| `salary`                | object    | Salary category: `low`, `medium`, or `high`                                |
| `quit`                  | int       | Target variable: 1 = employee left, 0 = employee stayed                     |

---

### 📈 Insights from the Data

* The majority of employees did **not receive a promotion** in the last 5 years.
* Most employees belong to the **sales department** and have a **low salary**.
* Employee **satisfaction level** appears to be a key factor affecting whether they leave.
* There is a wide range in **average monthly hours**, which may indicate overwork or imbalance in workload distribution.
* Only a small fraction of employees had work accidents or received promotions.

---
### 🔧 Recommended Preprocessing

To prepare the data for modeling, the following steps are recommended:

1. **Encode Categorical Features:**
   * Use One-Hot Encoding or Label Encoding for `department` and `salary`.

2. **Normalize or Scale Numerical Features (Optional):**
   * Standardize variables like `average_montly_hours`, `satisfaction_level` if using models sensitive to scale.

3. **Handle Class Imbalance (if needed):**
   * Check the distribution of the `quit` variable. If imbalanced, consider techniques like:
     * Stratified sampling
     * SMOTE (Synthetic Minority Over-sampling Technique)

4. **Feature Selection / Correlation Analysis:**
   * Analyze which features contribute most to employee churn using correlation heatmaps or feature importance scores.

---

### ✅ Why This Dataset Matters

Understanding employee churn is critical for businesses, as high turnover leads to:

* Increased recruitment and training costs
* Loss of experienced personnel
* Lower team morale and productivity
* Disruption in company operations

By analyzing this dataset and developing a reliable prediction model, HR departments can:

* Identify at-risk employees early
* Investigate and resolve churn drivers
* Improve employee satisfaction and retention strategies
* Make data-driven decisions for promotions, compensation, and workload balancing

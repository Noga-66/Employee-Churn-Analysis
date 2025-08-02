import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Initialize SHAP visualizations
shap.initjs()

# Load the pre-trained XGBoost model
model = joblib.load("best_xgb_model_for_deployment.pkl")

# Define the features used in the model
feature_columns = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'Work_accident',
    'promotion_last_5years', 'salary',
    'department_IT', 'department_RandD', 'department_accounting',
    'department_hr', 'department_management', 'department_marketing',
    'department_product_mng', 'department_sales', 'department_support',
    'department_technical', 'hours_level'
]
departments = ['IT', 'RandD', 'accounting', 'hr', 'management',
               'marketing', 'product_mng', 'sales', 'support', 'technical']

# Configure Streamlit page
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("👨‍💼 Employee Attrition Prediction Dashboard")

# Input form for employee data
with st.form("prediction_form"):
    st.subheader("📝 Enter Employee Data:")
    col1, col2, col3 = st.columns(3)
    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        number_project = st.number_input("Number of Projects", 1, 10, 3)
        Work_accident = st.selectbox("Work Accident", [0, 1])
        department = st.selectbox("Department", departments)
    with col2:
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
        average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 150)
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
        salary = st.selectbox("Salary Level (0=Low,1=Medium,2=High)", [0, 1, 2])
    with col3:
        time_spend_company = st.number_input("Years at Company", 1, 20, 3)
        hours_level = st.slider("Hours Level (Scaled)", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("🔮 Predict")

# Prediction logic after submitting the form
if submitted:
    # One-hot encoding for department
    department_encoded = [1 if department == d else 0 for d in departments]

    # Prepare input data for prediction
    input_data = np.array([satisfaction_level, last_evaluation, number_project,
                           average_montly_hours, time_spend_company, Work_accident,
                           promotion_last_5years, salary] + department_encoded + [hours_level]).reshape(1, -1)
    df_input = pd.DataFrame(input_data, columns=feature_columns)

    # Model prediction
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # Tabs for structured output
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Employee Input",
        "📌 Prediction Result",
        "🔍 SHAP Explanation",
        "📊 Visual Analysis"
    ])

    # Tab 1: Display input data and basic visualizations
    with tab1:
        st.subheader("📋 Entered Employee Data")
        st.dataframe(df_input.T.rename(columns={0: "Value"}))

        # Pie chart for binary features
        st.write("### 🧮 Binary Features")
        binary_features = {
            "Work Accident": Work_accident,
            "Promotion in 5Y": promotion_last_5years
        }
        fig_pie = px.pie(names=list(binary_features.keys()), values=list(binary_features.values()))
        st.plotly_chart(fig_pie)

        # Heatmap of input values
        st.write("### 🔥 Heatmap")
        fig_heat, ax = plt.subplots(figsize=(10, 1))
        sns.heatmap(df_input, cmap="YlGnBu", annot=True, fmt=".2f", cbar=False, ax=ax)
        st.pyplot(fig_heat)

        # Radar chart for selected inputs
        st.write("### 🧭 Radar Chart of Inputs")
        radar_features = ['satisfaction_level', 'last_evaluation', 'hours_level']
        radar_vals = df_input[radar_features].values.flatten().tolist()
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_features + [radar_features[0]],
            fill='toself'
        ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(radar_fig)

    # Tab 2: Prediction result
    with tab2:
        st.subheader("📌 Prediction Result")
        if prediction == 1:
            st.error("⚠ The employee is likely to leave the company.")
        else:
            st.success("✅ The employee is likely to stay at the company.")

        st.write(f"📊 Probability of Leaving: {proba:.2%}")

        # Gauge chart for probability
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            title={'text': "Probability of Leaving (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prediction == 1 else "green"},
                   'steps': [{'range': [0, 50], 'color': "lightgreen"},
                             {'range': [50, 100], 'color': "lightcoral"}]},
        ))
        st.plotly_chart(gauge)

    # Tab 3: SHAP explanation of prediction
    with tab3:
        st.subheader("🔍 SHAP Model Explanation")

        st.write("### 🌊 Waterfall Plot")
        fig_wf = plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             data=df_input.values[0],
                                             feature_names=feature_columns))
        st.pyplot(fig_wf)

        st.write("### 📶 Feature Importance (Bar Plot)")
        fig_bar, ax = plt.subplots()
        shap.bar_plot(shap_values[0], feature_names=feature_columns, max_display=10, show=False)
        st.pyplot(fig_bar)

    # Tab 4: Additional visual analysis
    with tab4:
        st.subheader("📊 Visual Analysis")

        # Beeswarm plot (duplicated input for visibility)
        st.write("### 🐝 SHAP Beeswarm (repeated 10 times for illustration)")
        repeated_df = pd.concat([df_input] * 10, ignore_index=True)
        repeated_shap = explainer.shap_values(repeated_df)
        fig_swarm = plt.figure()
        shap.summary_plot(repeated_shap, repeated_df, plot_type="dot", show=False)
        st.pyplot(fig_swarm)

        # Histogram for satisfaction
        st.write("### 📈 Histogram of Satisfaction Level")
        fig_hist, ax = plt.subplots()
        sns.histplot([satisfaction_level] * 10, bins=10, kde=True, ax=ax)
        st.pyplot(fig_hist)

        # Violin plot
        st.write("### 🎻 Violin Plot: Salary vs Satisfaction")
        violin_df = pd.DataFrame({"Salary": [salary] * 10, "Satisfaction": [satisfaction_level] * 10})
        fig_vio, ax = plt.subplots()
        sns.violinplot(x="Salary", y="Satisfaction", data=violin_df, ax=ax)
        st.pyplot(fig_vio)

        # Correlation heatmap
        st.write("### 🔗 Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 1))
        sns.heatmap(df_input, cmap="coolwarm", annot=True, fmt=".2f", cbar=False)
        st.pyplot(fig_corr)

        # Box plot for numerical features
        st.write("### 📦 Box Plot of Key Features")
        box_df = pd.DataFrame({
            "Satisfaction Level": [satisfaction_level],
            "Last Evaluation": [last_evaluation],
            "Hours Level": [hours_level]
        })
        fig_box, ax = plt.subplots()
        sns.boxplot(data=box_df, orient="h", ax=ax)
        st.pyplot(fig_box)

        # Scatter plot
        st.write("### ⚡ Scatter: Last Evaluation vs Satisfaction")
        fig_scatter = px.scatter(
            x=[satisfaction_level], y=[last_evaluation],
            labels={"x": "Satisfaction Level", "y": "Last Evaluation"},
            title="Satisfaction vs Evaluation",
            size=[number_project],
            color_discrete_sequence=["blue"]
        )
        st.plotly_chart(fig_scatter)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Dashboard")
st.markdown("""
Explore customer behavior, churn probability, and subscription patterns interactively.
""")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ChurnSpreadsheet.csv")
    return df

df = load_data()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filter Customers")

contract_filter = st.sidebar.multiselect(
    "Contract Type", df['ContractType'].unique(), default=df['ContractType'].unique()
)
gender_filter = st.sidebar.multiselect(
    "Gender", df['Gender'].unique(), default=df['Gender'].unique()
)
senior_filter = st.sidebar.multiselect(
    "Senior Citizen", df['SeniorCitizen'].unique(), default=df['SeniorCitizen'].unique()
)

# Apply filters
filtered_df = df[
    (df['ContractType'].isin(contract_filter)) &
    (df['Gender'].isin(gender_filter)) &
    (df['SeniorCitizen'].isin(senior_filter))
]

st.write(f"Showing {filtered_df.shape[0]} customers after filtering")

# ----------------------------
# KPI Cards
# ----------------------------
total_customers = filtered_df.shape[0]
churn_rate = (filtered_df['Churn'] == 'Yes').mean() * 100
avg_monthly_charge = filtered_df['MonthlyCharges'].mean()
long_term_customers = (filtered_df['TenureMonths'] > 60).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", total_customers)
col2.metric("💔 Churn Rate", f"{churn_rate:.1f}%")
col3.metric("💵 Avg Monthly Charge", f"${avg_monthly_charge:.2f}")
col4.metric("⏳ Long-Term Customers", long_term_customers)

st.markdown("---")

# ----------------------------
# Define chart layout defaults for readability
# ----------------------------
def apply_layout(fig):
    fig.update_layout(
        plot_bgcolor='#F9F9F9',
        paper_bgcolor='#F9F9F9',
        xaxis=dict(title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        yaxis=dict(title_font=dict(color="#333333"), tickfont=dict(color="#333333")),
        legend_font_color="#333333"
    )
    return fig


# ----------------------------
# Top At-Risk Customers (Interactive)
# ----------------------------
st.subheader("⚠️ Top At-Risk Customers")

# Generate mock ChurnProbability if missing or invalid
if 'ChurnProbability' not in filtered_df.columns or filtered_df['ChurnProbability'].isnull().any():
    filtered_df['ChurnProbability'] = np.clip(
        0.4 * (filtered_df['ContractType'] == 'Month-to-Month').astype(int) +
        0.3 * (filtered_df['TenureMonths'] < 12).astype(int) -
        0.05 * ((filtered_df[['TechSupport', 'OnlineBackup', 'StreamingTV', 'DeviceProtection']] == 'Yes').sum(axis=1)),
        0, 1
    )

# Slider for number of top-risk customers
top_n = st.slider("Number of top-risk customers to display", min_value=5, max_value=20, value=10)

# Select top N at-risk customers
top_risk = filtered_df.sort_values(by='ChurnProbability', ascending=False).head(top_n)

# Display with colored progress bars
for idx, row in top_risk.iterrows():
    prob = row['ChurnProbability']

    # Determine color
    if prob >= 0.7:
        color = "#FF4C4C"  # red
    elif prob >= 0.4:
        color = "#FFA500"  # orange
    else:
        color = "#4ECDC4"  # teal/green

    st.markdown(
        f"**Customer ID: {row['CustomerID']}** — Contract: {row['ContractType']}, Tenure: {row['TenureMonths']} months")

    # Colored progress bar using HTML/CSS
    st.markdown(
        f"""
        <div style='background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px;'>
            <div style='width: {prob * 100}%; background-color: {color}; height: 100%; border-radius: 5px; text-align: right; padding-right: 5px; color: white; font-weight: bold;'>
                {int(prob * 100)}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Churn Distribution Pie Chart
# ----------------------------
st.subheader("Churn Distribution")
churn_counts = filtered_df['Churn'].value_counts().reset_index()
churn_counts.columns = ['Churn', 'Count']

fig_churn = px.pie(
    churn_counts, names='Churn', values='Count', color='Churn',
    color_discrete_map={'Yes':'#FF6B6B','No':'#4ECDC4'},
    title="Churn vs Retained Customers"
)
apply_layout(fig_churn)
st.plotly_chart(fig_churn, use_container_width=True)

# ----------------------------
# Tenure Histogram
# ----------------------------
st.subheader("Customer Tenure Distribution by Churn")
fig_tenure = px.histogram(
    filtered_df, x='TenureMonths', nbins=30, color='Churn',
    color_discrete_map={'Yes':'#FF6B6B','No':'#4ECDC4'},
    labels={'TenureMonths':'Tenure (Months)'}
)
apply_layout(fig_tenure)
st.plotly_chart(fig_tenure, use_container_width=True)

# ----------------------------
# Average Monthly Charges by Contract Type
# ----------------------------
st.subheader("Average Monthly Charges by Contract Type")
avg_charges = filtered_df.groupby('ContractType')['MonthlyCharges'].mean().reset_index()
fig_avg_charges = px.bar(
    avg_charges, x='ContractType', y='MonthlyCharges', color='ContractType',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    labels={'MonthlyCharges':'Avg Monthly Charge'}, title="Avg Monthly Charges by Contract"
)
apply_layout(fig_avg_charges)
st.plotly_chart(fig_avg_charges, use_container_width=True)

# ----------------------------
# Add-On Adoption Rates
# ----------------------------
st.subheader("Add-On Adoption Rates")
add_ons = ['TechSupport','OnlineBackup','StreamingTV','DeviceProtection']
add_on_rates = filtered_df[add_ons].apply(lambda x: (x=='Yes').mean()).reset_index()
add_on_rates.columns = ['AddOn','AdoptionRate']

fig_addons = px.bar(
    add_on_rates, x='AddOn', y='AdoptionRate', text='AdoptionRate',
    color='AddOn', color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'AdoptionRate':'Rate'}, title="Add-On Adoption Rates"
)
fig_addons.update_traces(texttemplate='%{text:.1%}', textposition='outside')
apply_layout(fig_addons)
st.plotly_chart(fig_addons, use_container_width=True)

# ----------------------------
# Interactive Scatter: Tenure vs MonthlyCharges
# ----------------------------
st.subheader("Tenure vs Monthly Charges")
size_column = 'ChurnProbability' if 'ChurnProbability' in df.columns else None

fig_scatter = px.scatter(
    filtered_df, x='MonthlyCharges', y='TenureMonths',
    color='Churn', size=size_column,
    color_discrete_map={'Yes':'#FF6B6B','No':'#4ECDC4'},
    hover_data=['ContractType','PaymentMethod','TechSupport','StreamingTV'],
    labels={'MonthlyCharges':'Monthly Charge','TenureMonths':'Tenure (Months)'},
    title="Tenure vs Monthly Charges by Churn"
)
apply_layout(fig_scatter)
st.plotly_chart(fig_scatter, use_container_width=True)

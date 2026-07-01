import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Segmentation App - JPMC Task 2")

uploaded_file = st.file_uploader("Upload Mall_Customers.csv here", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Only use numeric columns for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = StandardScaler().fit_transform(X)

    k = st.slider("Select number of clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(X_scaled)

    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', palette='tab10', ax=ax)
    st.pyplot(fig)
    
    # FIXED: Only show mean for numeric columns
    st.write("### Segment Averages")
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    st.dataframe(df.groupby('Segment')[numeric_cols].mean().round(2))
else:
    st.info("👆 Upload `Mall_Customers.csv` to run the app")

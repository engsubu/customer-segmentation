import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Customer Segmentation App")

df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)

k = st.slider("Select number of clusters", 2, 10, 5)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', ax=ax)
st.pyplot(fig)
st.dataframe(df.groupby('Segment').mean())

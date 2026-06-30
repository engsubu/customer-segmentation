# Customer Segmentation Project - JPMC Forage Task 2

## Objective
Segment customers based on behavior and demographics using KMeans clustering.

## Dataset
Mall Customer Segmentation Data: Age, Annual Income, Spending Score

## Approach
1. Data Preprocessing: Scaled features with StandardScaler
2. Clustering: KMeans with k=5 clusters 
3. Visualization: Scatter plot of Income vs Spending Score by Segment

## Key Insights
Cluster 0: High Income, High Spenders -> Target for premium products
Cluster 1: High Income, Low Spenders -> Careless/Need offers
Cluster 2: Low Income, High Spenders -> Budget but engaged
Cluster 3: Low Income, Low Spenders -> Standard customers
Cluster 4: Medium Income, Medium Spenders -> Sensible

## How to Run
pip install -r requirements.txt
streamlit run app.py

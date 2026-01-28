# 🟢 Customer Segmentation K-Means Dashboard

This project implements an **interactive Customer Segmentation Dashboard** using **K-Means Clustering** to group customers based on their purchasing behavior and similarities.

The application is built using **Streamlit** and allows users to dynamically explore how different features and cluster settings affect customer segmentation.

---

## 🚀 Project Objective

In many businesses, all customers are treated the same, which can lead to:
- Inefficient inventory planning  
- Poor marketing strategies  
- Missed upselling opportunities  

This project aims to:
👉 **Discover hidden customer groups without predefined labels**  
👉 Help businesses make **data-driven decisions** using customer purchasing patterns

---

## 🧠 Clustering Approach

- **Algorithm Used:** K-Means Clustering  
- **Type:** Unsupervised Learning  
- **Distance Metric:** Euclidean Distance  

### Features Used
The clustering is based on numerical purchasing behavior features:
- Fresh  
- Milk  
- Grocery  
- Frozen  
- Detergents_Paper  
- Delicassen  

All features are **standardized** to ensure fair distance-based clustering.

---

## 🖥️ Application Features

### 🔧 Clustering Controls
Users can interactively:
- Select **two numerical features** for visualization
- Choose the **number of clusters (K)** using a slider (2–10)
- Set a **random state** (optional)
- Explicitly run clustering using a **Run Clustering** button

This design helps users understand that **clustering is an action**, and results change with different inputs.

---

### 📊 Visualization
- 2D scatter plot of selected features
- Different colors represent different clusters
- Cluster centers are clearly marked

---

### 📋 Cluster Summary
A summary table shows:
- Cluster ID  
- Number of customers in each cluster  
- Average values of selected features  

This makes the clusters easier to interpret for business users.

---

### 💡 Business Interpretation
Each cluster is described in **simple, non-technical language**, for example:
- High-spending customers
- Budget-conscious customers
- Moderate or selective spenders  

This bridges the gap between **machine learning output and business insight**.

---

### ℹ️ User Guidance
The app includes guidance to help users understand clustering results:
> Customers in the same cluster exhibit similar purchasing behaviour and can be targeted with similar business strategies.

---

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

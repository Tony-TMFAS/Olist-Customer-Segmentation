
# 🛍️ Olist Customer Segmentation Project

This project applies **machine learning** to real e-commerce data from the Brazilian marketplace **Olist** to **segment customers** based on their purchasing behavior. It combines **data science**, **API development**, and **web app deployment** into one smooth pipeline — ideal for portfolio-building or production-ready insights.

---

## 🚀 What This Project Does

We segment Olist’s customers using **RFM Analysis** (Recency, Frequency, and Monetary Value) and **K-Means Clustering** to uncover valuable patterns, such as:

- 🧊 Low-value customers  
- 🔥 High-value customers  
- ⏳ At-risk or dormant customers  
- 🆕 New buyers

Users can input RFM metrics into a **Streamlit app**, which communicates with a **FastAPI backend** that runs the trained model and returns the predicted customer segment.

---

## 🛠️ Tools & Technologies

| Tool         | Purpose                                   |
|--------------|--------------------------------------------|
| **Pandas**   | Data cleaning and manipulation             |
| **Matplotlib** | Visualization of customer clusters       |
| **Scikit-learn** | K-Means Clustering & StandardScaler    |
| **FastAPI**  | Lightweight API for model serving          |
| **Streamlit**| Interactive frontend web app               |
| **Render**   | Deploying the FastAPI backend online       |
| **Streamlit Cloud** | Deploying the Streamlit frontend    |
| **Git & GitHub** | Version control and project hosting    |

---

## 📦 Folder Structure

```
.
├── main.py                 # FastAPI backend code
├── streamlit_app.py        # Frontend Streamlit app
├── kmeans_model.pkl        # Trained clustering model
├── scaler.pkl              # Scaler used to normalize RFM features
├── clustered_rfm.csv       # CSV with original data and predicted clusters
├── requirements.txt        # Required Python libraries
├── README.md               # This file 😊
└── olist_data/             # Unzipped Olist datasets
```

---

## 📊 Step-by-Step Breakdown

### 1. **Data Preparation**

- Unzipped `olist_data.zip`, which contains CSVs like:
  - `olist_orders_dataset.csv`
  - `olist_customers_dataset.csv`
  - `olist_order_items_dataset.csv`

- Loaded datasets using `pandas.read_csv`.

- Merged relevant CSVs using `customer_id` and `order_id`.

---

### 2. **Feature Engineering: RFM Table**

For each unique customer, we calculated:

| Feature   | Meaning                                  |
|-----------|-------------------------------------------|
| Recency   | Days since last purchase                  |
| Frequency | Total number of orders                    |
| Monetary  | Total amount spent (sum of item prices)   |

```python
rfm = full_data.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (today - x.max()).days,
    'order_id': 'nunique',
    'price': 'sum'
}).reset_index()
```

---

### 3. **Data Cleaning**

- Removed canceled orders.
- Handled missing values.
- Rounded monetary values to remove centavos.

---

### 4. **Clustering: K-Means**

- Scaled `recency`, `frequency`, and `monetary` using `StandardScaler`.

- Determined the optimal number of clusters using the **elbow method**.

- Trained a K-Means model:

```python
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
```

- Saved the model and scaler:

```python
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

### 5. **FastAPI Backend**

- Created an API with FastAPI to accept RFM input and return the predicted segment.

```python
@app.post("/predict")
def predict_segment(customer: Customer):
    ...
```

- Hosted on **Render**: [render.com](https://render.com)

---

### 6. **Streamlit Frontend**

- Built an interactive UI to:
  - Accept RFM input
  - Send it to FastAPI
  - Display the predicted segment
  - Visualize cluster distributions

- Read the FastAPI URL securely using Streamlit secrets:

```python
api_url = st.secrets["API_URL"]
```

- Hosted on **Streamlit Cloud**: [streamlit.io/cloud](https://streamlit.io/cloud)

---

## 📈 Visualizations

- **Bar chart** showing the number of customers per cluster
- Optional: Add pie charts, scatter plots, or line trends using `matplotlib` or `seaborn`

---

## 🌐 Deployment Steps

1. **Push code to GitHub**
2. **Deploy FastAPI** on [Render](https://render.com) using:
   - `main.py`
   - `kmeans_model.pkl` and `scaler.pkl`
3. **Deploy Streamlit** on [Streamlit Cloud](https://streamlit.io/cloud):
   - Use `streamlit_app.py`
   - Set secret in Settings → Secrets:

     ```toml
     API_URL = "https://your-fastapi-url.onrender.com/predict"
     ```

---

## ✅ To Run Locally

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **Run FastAPI**

```bash
uvicorn main:app --reload
```

3. **Run Streamlit**

```bash
streamlit run streamlit_app.py
```

---

## 🧠 What You Learn

- How to transform raw e-commerce data into actionable clusters.
- How to build, train, and deploy a K-Means clustering model.
- How to create a real-world ML product using APIs and web UIs.
- How to combine multiple modern tools in a full data-to-deployment workflow.

---


## 🌐 Live App

👉 Click here to try the Streamlit App: https://olist-seg.streamlit.app/


## 🙌 Credits

- Dataset from [Olist e-commerce public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Streamlit + FastAPI community docs

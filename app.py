import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fcmeans import FCM
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Fungsi untuk menghitung skor RFM
def calculate_rfm(df):
    # Tampilkan nama kolom untuk debugging
    
    # Pastikan kolom yang dibutuhkan ada dalam DataFrame
    if all(col in df.columns for col in ['CustomerID', 'InvoiceDate', 'TotalPrice']):
        rfm_df = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (pd.Timestamp.now() - pd.to_datetime(x).max()).days,
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'
        })
        rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
        return rfm_df
    else:
        st.error("Kolom yang dibutuhkan tidak ada dalam DataFrame.")
        return pd.DataFrame()

# Fungsi untuk normalisasi data
def normalize_data(df, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        st.error("Metode normalisasi tidak valid.")
        return df
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# Fungsi untuk menampilkan hasil clustering dan anggota cluster
def display_clusters(df, labels):
    df['Cluster'] = labels
    st.write("Cluster Summary")
    st.write(df.groupby('Cluster').mean())

    st.write("Cluster Members")
    for cluster in sorted(df['Cluster'].unique()):
        st.write(f"Cluster {cluster}")
        st.write(df[df['Cluster'] == cluster])

# Fungsi untuk melakukan clustering
def clustering(df, method, n_clusters):
    if method == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=0)
        labels = model.fit_predict(df)
    elif method == 'Fuzzy C-Means':
        model = FCM(n_clusters=n_clusters)
        model.fit(df)
        labels = model.predict(df)
    else:
        st.error("Metode clustering tidak valid.")
        return None

    return labels

# Fungsi untuk plot Elbow
def plot_elbow(data):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    st.pyplot(plt)

# Fungsi untuk plot Silhouette
def plot_silhouette(data):
    silhouette_avg = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_avg.append(score)
    plt.figure()
    plt.plot(range(2, 11), silhouette_avg, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    st.pyplot(plt)

# Fungsi utama aplikasi
def main():
    st.set_page_config(page_title="Market Research Analysis", layout="wide")

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Upload File", "RFM Analysis", "Clustering"],
            icons=["cloud-upload", "bar-chart-line", "diagram-3"],
            menu_icon="house",
            default_index=0,
        )

    if selected == "Upload File":
        st.title("Upload Your Excel File")
        uploaded_file = st.file_uploader('Upload your Excel file', type=['xlsx', 'xls'])
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.subheader('Data Preview')
            st.write(df)

    elif selected == "RFM Analysis":
        if "df" in st.session_state:
            df = st.session_state.df
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
            rfm_df = calculate_rfm(df)
            
            if not rfm_df.empty:
                st.session_state.rfm_df = rfm_df
                st.subheader('RFM Data')
                st.write(rfm_df)
                
                metrics = st.multiselect('Select RFM Metrics for Analysis', ['Recency', 'Frequency', 'Monetary'], default=['Recency', 'Frequency', 'Monetary'])
                
                if metrics:
                    rfm_selected = rfm_df[metrics]
                    rfm_scaled = normalize_data(rfm_selected, method='minmax')
                    
                    st.subheader('Normalized RFM Data (Min-Max)')
                    st.write(pd.DataFrame(rfm_scaled, columns=metrics))
                    
                    st.subheader('Elbow Method')
                    plot_elbow(rfm_scaled)
                    
                    st.subheader('Silhouette Method')
                    plot_silhouette(rfm_scaled)
        else:
            st.warning("Please upload a file first.")

    elif selected == "Clustering":
        if "rfm_df" in st.session_state:
            metrics = st.multiselect('Select RFM Metrics for Clustering', ['Recency', 'Frequency', 'Monetary'], default=['Recency', 'Frequency', 'Monetary'])
            
            if metrics:
                rfm_selected = st.session_state.rfm_df[metrics]
                rfm_scaled = normalize_data(rfm_selected, method='minmax')
                
                method = st.selectbox('Select Clustering Method', ['K-Means', 'Fuzzy C-Means'])
                n_clusters = st.slider('Number of Clusters', 2, 10, 3)
                
                if st.button('Perform Clustering'):
                    labels = clustering(rfm_scaled, method, n_clusters)
                    if labels is not None:
                        st.subheader('Clustered RFM Table')
                        display_clusters(st.session_state.rfm_df, labels)
                        
                        if method == 'K-Means':
                            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(rfm_scaled)
                            sse = kmeans.inertia_
                            dbi = davies_bouldin_score(rfm_scaled, labels)
                            
                            st.subheader('Clustering Validity Metrics')
                            st.write(f'SSE (Sum of Squared Errors): {sse}')
                            st.write(f'DBI (Davies-Bouldin Index): {dbi}')
                        elif method == 'Fuzzy C-Means':
                            st.info("Fuzzy C-Means clustering is selected. Validity metrics calculation for FCM is not implemented.")
        else:
            st.warning("Please upload a file first.")

if __name__ == "__main__":
    main()

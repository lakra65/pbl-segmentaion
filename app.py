import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

with open('kmeans_model.pkl', 'rb') as file: 
    modele =pickle.load(file)  
   

# add title
st.title('Data Analysis Application')
st.subheader('This is a simple data analysis application for customer segmentation')

# create a dropdown list to choose a dataset
dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset', dataset_options)

# load the selected dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')

# button to upload custom dataset
uploaded_file = st.file_uploader('Upload a custom dataset', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # process the uploaded file
    df = pd.read_csv(uploaded_file)  # assuming the uploaded file is in CSV format

# display the dataset
st.write(df)

# display the number of Rows and Column from the selected data
st.write('Number of Rows:', df.shape[0])
st.write('Number of Columns:', df.shape[1])

# display the column names of selected data with their data types
st.write('Column Names and Data Types:', df.dtypes)

# print the null values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null Values')

# display the summary statistics of the selected data
st.write('Summary Statistics:', df.describe())

# Create a pairplot
st.subheader('Pairplot')
# select the column to be used as hue in pairplot
hue_column = st.selectbox('Select a column to be used as hue', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

# Create a heatmap
st.subheader('Heatmap')
# select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()

from plotly import graph_objects as go

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                       x=corr_matrix.columns,
                                       y=corr_matrix.columns,
                                       colorscale='Viridis'))
st.plotly_chart(heatmap_fig)


# Set the title of the app
st.title("Clustering Plot with Streamlit")

# Sidebar for user input
st.sidebar.header("User  Input Parameters")
num_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
# Display the DataFrame
st.write("Here is the sample data:")
st.dataframe(df)

# Feature selection
st.sidebar.header("Select Features")
features = st.sidebar.multiselect("Choose features to display", options=df.columns.tolist())

# Display selected features
if features:
    st.write("You selected the following features:")
    st.dataframe(df[features])
    st.pyplot()
else:
    st.write("Please select at least one feature to display.")
# Generate synthetic data
X, y = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=1.0, random_state=42)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Create a DataFrame for visualization

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df['Cluster'] = y_kmeans

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(df['Feature 1'], df['Feature 2'], c=df['Cluster'], cmap='viridis', marker='o')
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
ax.set_title("KMeans Clustering")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "httpx",
#   "scikit-learn"
# ]
# ///

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Prompt user to input OpenAI API key manually
AIPROXY_TOKEN = input("Please enter your OpenAI API key: ").strip()
os.environ["AIPROXY_TOKEN"] = AIPROXY_TOKEN
openai.api_key = AIPROXY_TOKEN

def load_csv(file_path):
    """Load CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def perform_eda(df):
    """Perform Exploratory Data Analysis (EDA) on the DataFrame."""
    eda_results = {
        "shape": df.shape,
        "columns": df.columns.to_list(),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe().to_dict(),
    }
    return eda_results

def detect_outliers(df):
    """Detect outliers in the DataFrame using Isolation Forest."""
    try:
        numeric_columns = df.select_dtypes(include=[np.number])
        if not numeric_columns.empty:
            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            df['outlier'] = isolation_forest.fit_predict(numeric_columns)
    except Exception as e:
        print(f"Error detecting outliers: {e}")

def perform_clustering(df):
    """Perform K-means clustering on the DataFrame."""
    try:
        numeric_columns = df.select_dtypes(include=[np.number])
        if not numeric_columns.empty:
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['cluster'] = kmeans.fit_predict(numeric_columns)
    except Exception as e:
        print(f"Error performing clustering: {e}")

def create_visualizations(df):
    """Create and save visualizations as PNG files."""
    try:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

        # Distribution plot for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns[:3]:  # Limit to first 3 for simplicity
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(f"distribution_{col}.png")
            plt.close()

        # Pairplot for first 3 numeric columns
        if len(numeric_columns) >= 2:
            sns.pairplot(df[numeric_columns[:3]])
            plt.savefig("pairplot.png")

        # Outliers plot (if outliers were detected)
        if 'outlier' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df.index, y=df[numeric_columns[0]], hue=df['outlier'], palette='coolwarm')
            plt.title("Outlier Detection (first numeric column vs index)")
            plt.savefig("outliers.png")
            plt.close()

        # Cluster plot (if clusters were formed)
        if 'cluster' in df.columns and len(numeric_columns) >= 2:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]], hue=df['cluster'], palette='viridis')
            plt.title("Cluster Analysis")
            plt.savefig("clusters.png")
            plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def generate_readme(eda_results):
    """Generate a README.md file with a narrative of the analysis."""
    try:
        with open("README.md", "w") as file:
            file.write("# Automated Data Analysis Report\n\n")
            file.write("## Data Overview\n\n")
            file.write(f"- Number of rows: {eda_results['shape'][0]}\n")
            file.write(f"- Number of columns: {eda_results['shape'][1]}\n")
            file.write("- Column names and data types:\n\n")
            for col, dtype in eda_results['data_types'].items():
                file.write(f"  - {col}: {dtype}\n")
            file.write("\n## Summary Statistics\n\n")
            for col, stats in eda_results['summary_statistics'].items():
                file.write(f"### {col}\n")
                for stat_name, value in stats.items():
                    file.write(f"- {stat_name}: {value}\n")
            file.write("\n## Missing Values\n\n")
            for col, missing in eda_results['missing_values'].items():
                file.write(f"- {col}: {missing} missing values\n")
            file.write("\n## Visualizations\n\n")
            file.write("![](correlation_heatmap.png)\n\n")
            for col in eda_results['columns'][:3]:
                file.write(f"![](distribution_{col}.png)\n\n")
            file.write("![](pairplot.png)\n\n")
            file.write("![](outliers.png)\n\n")
            file.write("![](clusters.png)\n\n")
    except Exception as e:
        print(f"Error generating README.md: {e}")

def main(file_path):
    """Main function to run the analysis pipeline."""
    df = load_csv(file_path)
    if df is not None:
        eda_results = perform_eda(df)
        detect_outliers(df)
        perform_clustering(df)
        create_visualizations(df)
        generate_readme(eda_results)

if __name__ == "__main__":
    from google.colab import files
    uploaded = files.upload()  # Prompt for file upload in Colab
    file_path = list(uploaded.keys())[0]
    main(file_path)
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Prompt user to input OpenAI API key manually
AIPROXY_TOKEN = input("Please enter your OpenAI API key: ").strip()
os.environ["AIPROXY_TOKEN"] = AIPROXY_TOKEN
openai.api_key = AIPROXY_TOKEN


def create_output_directories(dataset_name):
    """Create directory for storing README and PNG files for the given dataset."""
    dataset_dir = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def load_csv(file_path):
    """Load CSV file into a DataFrame with error handling for encoding issues."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"CSV file '{file_path}' loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV file '{file_path}': {e}")
        return None


def handle_missing_values(df):
    """Handle missing values by imputing with the mean for numeric columns."""
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        print("Missing values imputed successfully.")
    except Exception as e:
        print(f"Error handling missing values: {e}")


def remove_non_numeric_columns(df):
    """Remove non-numeric columns from the DataFrame."""
    try:
        df = df.select_dtypes(include=[np.number])
        print("Non-numeric columns removed.")
    except Exception as e:
        print(f"Error removing non-numeric columns: {e}")
    return df


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


def create_visualizations(df, output_dir):
    """Create and save visualizations as PNG files in the dataset directory."""
    try:
        sns.set_theme(style="darkgrid")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns[:3]:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
            plt.close()

        if len(numeric_columns) >= 2:
            sns.pairplot(df[numeric_columns[:3]])
            plt.savefig(os.path.join(output_dir, "pairplot.png"))

        if 'outlier' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df.index, y=df[numeric_columns[0]], hue=df['outlier'], palette='coolwarm')
            plt.title("Outlier Detection")
            plt.savefig(os.path.join(output_dir, "outliers.png"))
            plt.close()

    except Exception as e:
        print(f"Error creating visualizations: {e}")


def generate_readme(eda_results, output_dir):
    """Generate a README.md file with the analysis summary."""
    try:
        with open(os.path.join(output_dir, "README.md"), "w") as file:
            file.write("# Automated Data Analysis Report\n\n")
            file.write(f"**Rows:** {eda_results['shape'][0]}\n")
            file.write(f"**Columns:** {eda_results['shape'][1]}\n")
            file.write("\n## Column Info\n")
            for col, dtype in eda_results['data_types'].items():
                file.write(f"- {col}: {dtype}\n")
            file.write("\n## Summary Statistics\n")
            for col, stats in eda_results['summary_statistics'].items():
                file.write(f"### {col}\n")
                for stat_name, value in stats.items():
                    file.write(f"- {stat_name}: {value}\n")
    except Exception as e:
        print(f"Error generating README.md: {e}")


def main(file_path, dataset_name):
    output_dir = create_output_directories(dataset_name)
    df = load_csv(file_path)
    if df is not None:
        handle_missing_values(df)
        df = remove_non_numeric_columns(df)
        eda_results = perform_eda(df)
        detect_outliers(df)
        perform_clustering(df)
        create_visualizations(df, output_dir)
        generate_readme(eda_results, output_dir)


if __name__ == "__main__":
    datasets = ["goodreads.csv", "happiness.csv", "media.csv"]
    for dataset in datasets:
        main(dataset, dataset.split('.')[0])

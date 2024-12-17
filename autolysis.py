import os
import pandas as pd
import numpy as np
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

def create_project_directory(file_path):
    """Create a directory named after the file (without extension) to store analysis results."""
    directory_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        os.makedirs(directory_name, exist_ok=True)
        print(f"Directory '{directory_name}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{directory_name}': {e}")
    return directory_name

def load_csv(file_path):
    """Load CSV file into a DataFrame with error handling for encoding issues."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("CSV file loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
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
    """Create and save visualizations as PNG files in the project directory."""
    try:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

        # Distribution plot for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns[:3]:  # Limit to first 3 for simplicity
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
            plt.close()

        # Pairplot for first 3 numeric columns
        if len(numeric_columns) >= 2:
            sns.pairplot(df[numeric_columns[:3]])
            plt.savefig(os.path.join(output_dir, "pairplot.png"))

        if 'outlier' in df.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df.index, y=df[numeric_columns[0]], hue=df['outlier'], palette='coolwarm')
            plt.title("Outlier Detection")
            plt.savefig(os.path.join(output_dir, "outliers.png"))
            plt.close()

        if 'cluster' in df.columns and len(numeric_columns) >= 2:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]], hue=df['cluster'], palette='viridis')
            plt.title("Cluster Analysis")
            plt.savefig(os.path.join(output_dir, "clusters.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def generate_readme(eda_results, output_dir):
    """Generate a README.md file in the project directory."""
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as file:
            file.write("# Data Analysis Report\n\n")
            file.write("## Data Overview\n\n")
            file.write(f"**Number of rows**: {eda_results['shape'][0]}\n\n")
            file.write(f"**Number of columns**: {eda_results['shape'][1]}\n\n")
            file.write("### Column Details\n")
            for col, dtype in eda_results['data_types'].items():
                file.write(f"- **{col}**: {dtype}\n")
            
            file.write("\n## Visualizations\n\n")
            file.write("![](correlation_heatmap.png)\n\n")
            for col in eda_results['columns'][:3]:
                file.write(f"![](distribution_{col}.png)\n\n")
            file.write("![](pairplot.png)\n\n")
            file.write("![](outliers.png)\n\n")
            file.write("![](clusters.png)\n\n")
    except Exception as e:
        print(f"Error generating README.md: {e}")

def generate_openai_summary(file_path):
    """Use OpenAI API to generate a summary of the analysis process."""
    try:
        with open(file_path, "r") as file:
            content = file.read()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes Python scripts."},
                {"role": "user", "content": f"Provide a summary of the following script:\n\n{content}"}
            ],
            max_tokens=150
        )
        
        summary = response.choices[0].message['content'].strip()
        print("OpenAI Summary:\n", summary)
    except Exception as e:
        print(f"Error generating summary with OpenAI API: {e}")


def main(file_path):
    """Main function to run the analysis pipeline."""
    output_dir = create_project_directory(file_path)
    df = load_csv(file_path)
    if df is not None:
        handle_missing_values(df)
        df = remove_non_numeric_columns(df)
        eda_results = perform_eda(df)
        detect_outliers(df)
        perform_clustering(df)
        create_visualizations(df, output_dir)
        generate_readme(eda_results, output_dir)
        generate_openai_summary(file_path)

if __name__ == "__main__":
    file_path = input('Enter the file path to your CSV: ').strip('"').strip("'")
    main(file_path)
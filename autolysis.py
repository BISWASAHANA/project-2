# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "openai", 
#   "httpx"
# ]
# ///

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import httpx

# Check for AI Proxy token
AIPROXY_TOKEN = os.environ.get('AIPROXY_TOKEN')
if not AIPROXY_TOKEN:
    raise EnvironmentError('AIPROXY_TOKEN environment variable is not set')

# Function to read the CSV file
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit(1)

# Function to perform exploratory data analysis (EDA)
def perform_eda(df):
    analysis = {}
    analysis['summary_statistics'] = df.describe().to_dict()
    analysis['missing_values'] = df.isnull().sum().to_dict()
    analysis['data_types'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    return analysis

# Function to create visualizations
def create_visualizations(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Example Visualization 1: Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    # Example Visualization 2: Distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue')
        distplot_path = os.path.join(output_dir, 'distribution_plot.png')
        plt.savefig(distplot_path)
        plt.close()
    
    # Example Visualization 3: Pairplot (if not too large)
    if len(numeric_columns) > 1:
        sns.pairplot(df[numeric_columns[:4]])  # Limit to 4 columns for performance
        pairplot_path = os.path.join(output_dir, 'pairplot.png')
        plt.savefig(pairplot_path)
        plt.close()

# Function to generate a README.md file
def generate_readme(analysis, output_dir):
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis\n\n")
        f.write("## Data Overview\n\n")
        f.write("**Summary Statistics:**\n\n")
        f.write(pd.DataFrame(analysis['summary_statistics']).to_markdown(index=True) + "\n\n")
        
        f.write("**Missing Values:**\n\n")
        f.write(pd.DataFrame(analysis['missing_values'], index=["Missing Values"]).to_markdown(index=True) + "\n\n")
        
        f.write("**Data Types:**\n\n")
        f.write(pd.DataFrame(analysis['data_types'], index=["Data Type"]).to_markdown(index=True) + "\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("1. ![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("2. ![Distribution Plot](distribution_plot.png)\n")
        f.write("3. ![Pairplot](pairplot.png)\n")
    print(f"README.md created at {readme_path}")

# Main function to run the analysis
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <csv_filename>")
        exit(1)
    
    csv_filename = sys.argv[1]
    output_dir = os.path.splitext(os.path.basename(csv_filename))[0]
    
    # Step 1: Load the CSV
    df = load_csv(csv_filename)
    
    # Step 2: Perform EDA
    analysis = perform_eda(df)
    
    # Step 3: Create Visualizations
    create_visualizations(df, output_dir)
    
    # Step 4: Generate README.md
    generate_readme(analysis, output_dir)

if __name__ == "__main__":
    main()

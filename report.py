import pandas as pd
import sys
import os

def analyze_raw_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        # Read CSV (handling potential formatting issues)
        df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
        
        # Check if AUC column exists
        if 'auc_test' not in df.columns:
            print("Column 'auc_test' not found in this file.")
            print(f"Available columns: {list(df.columns)}")
            return

        # Ensure data is numeric
        df['best_auc'] = pd.to_numeric(df['auc_test'], errors='coerce')
        df['best_fitness'] = pd.to_numeric(df['acc_test'], errors='coerce')
        
        # Find row with Maximum AUC
        best_idx = df['best_auc'].idxmax()
        best_row = df.loc[best_idx]
        
        print("\n" + "="*40)
        print(f"ANALYSIS: {os.path.basename(file_path)}")
        print("="*40)
        print(f"Generation:           {best_row['generation']}")
        print("-" * 40)
        print(f"🏆 BEST AUC:          {best_row['best_auc']:.4f}")
        print(f"✅ Matching Fitness:  {best_row['best_fitness']:.4f} (Accuracy)")
        print("="*40 + "\n")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report.py <path_to_csv_file>")
    else:
        analyze_raw_file(sys.argv[1])
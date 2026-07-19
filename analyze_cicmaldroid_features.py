import pandas as pd
import os

def analyze_dataset(csv_path):
    print("=" * 50)
    print("TASK 1: DATASET OVERVIEW")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return None, None
        
    df = pd.read_csv(csv_path)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1] - 1}") # Excluding label
    
    label_col = 'Class' if 'Class' in df.columns else df.columns[-1]
    
    features = [col for col in df.columns if col != label_col]
    
    print("\nExample Feature Names (first 10):")
    for f in features[:10]:
        print(f" - {f}")
        
    print("\nData Types:")
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        print(f" - {dtype}: {count} columns")
        
    print(f"\nLabel Column: {label_col}")
    print(f"Unique Labels: {df[label_col].unique()}")
    
    return df, features

def categorize_features(features):
    print("\n" + "=" * 50)
    print("TASK 2: FEATURE CATEGORIZATION")
    print("=" * 50)
    
    categories = {
        'Permissions': [],
        'Intents/Actions': [],
        'API Calls / Methods': [],
        'System Calls': [],
        'Other / Binders': []
    }
    
    for f in features:
        f_lower = str(f).lower()
        if 'permission' in f_lower:
            categories['Permissions'].append(f)
        elif 'intent' in f_lower or 'action' in f_lower:
            categories['Intents/Actions'].append(f)
        elif 'ljava/' in f_lower or 'landroid/' in f_lower or 'invoke-' in f_lower:
            categories['API Calls / Methods'].append(f)
        elif f_lower.startswith('sys_') or f_lower.startswith('__'):
            categories['System Calls'].append(f)
        else:
            categories['Other / Binders'].append(f)
            
    for cat, items in categories.items():
        print(f"\nCategory Name: {cat}")
        print(f"Number of Features: {len(items)}")
        if items:
            print(f"Example Features: {items[:3]}")
            
    return categories

def extractability_analysis():
    print("\n" + "=" * 50)
    print("TASK 3: EXTRACTABILITY FROM APK")
    print("=" * 50)
    
    print("Permissions: YES (Extracted from AndroidManifest.xml)")
    print("Intents/Actions: YES (Extracted from AndroidManifest.xml)")
    print("API Calls / Methods: YES (Extracted from classes.dex via static analysis)")
    print("System Calls: NO / PARTIALLY (Requires dynamic execution/sandboxing to capture trace, static extraction is extremely difficult and often inaccurate)")
    print("Other / Binders: PARTIALLY (Depends on whether they are static strings or dynamic binder transactions)")

def tool_recommendations():
    print("\n" + "=" * 50)
    print("TASK 4: RECOMMENDED TOOLS FOR EXTRACTION")
    print("=" * 50)
    
    print("| Category             | Recommended Tool(s)                                |")
    print("|----------------------|----------------------------------------------------|")
    print("| Permissions          | Androguard, APKTool, AAPT, MobSF (Static)          |")
    print("| Intents/Actions      | Androguard, APKTool, AAPT, MobSF (Static)          |")
    print("| API Calls / Methods  | Androguard, JADX, MobSF (Static)                   |")
    print("| System Calls         | Cuckoo Sandbox, DroidBox, MobSF (Dynamic), Frida   |")

def direct_prediction_feasibility():
    print("\n" + "=" * 50)
    print("TASK 5: DIRECT PREDICTION FEASIBILITY")
    print("=" * 50)
    
    print("Determine whether the trained XGBoost model can directly predict uploaded APK files.")
    print("\nAnswer clearly:")
    print("NO")
    print("\nExplain why:")
    print("The XGBoost model only accepts a numerical feature vector of length exactly 470 (which consists of frequencies of specific syscalls and binders).")
    print("An APK is a compiled zip archive containing DEX files, resources, and a manifest. The model does not understand binary APK files.")

def preprocessing_pipeline():
    print("\n" + "=" * 50)
    print("TASK 6: REQUIRED PREPROCESSING PIPELINE")
    print("=" * 50)
    print("To predict an APK, you MUST convert it into the exact 470-dimensional numerical feature vector.")
    print("Since this dataset heavily relies on System Calls and Binders (frequency based), static analysis (like APKTool) is NOT ENOUGH.")
    print("You must:")
    print("1. Run the APK in an instrumented Android Sandbox (e.g., Cuckoo or MobSF dynamic).")
    print("2. Capture the exact frequency of the 470 specific system calls and binders.")
    print("3. Map those captured frequencies to the exact column indexes the XGBoost model expects.")

def architecture():
    print("\n" + "=" * 50)
    print("TASK 7: FEATURE EXTRACTION ARCHITECTURE")
    print("=" * 50)
    print("Upload APK")
    print("    v")
    print("Dynamic Sandbox Execution (DroidBox/MobSF/Frida)")
    print("    v")
    print("Trace Capture (Syscalls & Binders)")
    print("    v")
    print("Frequency Calculation (Count occurrences)")
    print("    v")
    print("Feature Vector Generation (Map to 470 columns, pad with 0s for missing)")
    print("    v")
    print("Scaling (Apply scaler.pkl)")
    print("    v")
    print("XGBoost Prediction")
    print("    v")
    print("Result Generation")

if __name__ == "__main__":
    csv_path = r"C:\final year project\IDS_Project\dataset\CICMalDroid2020\CSV\feature_vectors_syscallsbinders_frequency_5_Cat.csv"
    
    df, features = analyze_dataset(csv_path)
    if df is not None:
        categories = categorize_features(features)
        extractability_analysis()
        tool_recommendations()
        direct_prediction_feasibility()
        preprocessing_pipeline()
        architecture()
        print("\nAnalysis complete. Run successful.")

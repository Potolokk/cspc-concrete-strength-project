import pandas as pd
from pathlib import Path


raw_path = Path("Concrete_Data.csv")
clean_path = Path("Clean_Data.csv")
summ_path = Path("Data_Quality_Summary.txt")

def load_data(path=raw_path):
    '''Loads the raw csv file into a data frame'''
    path = Path(path) #converts to a Path object, as the argument may arrive as a plain text
    if not path.exists():
        raise FileNotFoundError(f"file not found {path}")
    return pd.read_csv(path) 

def check_missing(df):
    '''Returns the number of missing values per column'''
    return df.isna().sum()

def check_duplicates(df):
    '''Returns the number of fully identical rows in DF'''
    return int(df.duplicated().sum()) #converts to integer for a cleaner behaviour

def check_types(df):
    '''Returns the data type of each column'''
    return df.dtypes #catches if a number column was read as a text

def find_outliers(df, factor = 1.5):
    '''Returns the number of outliers per column using IQR (interquartile range) method'''
    num = df.select_dtypes("number")
    Q1,Q3 = num.quantile(0.25),num.quantile(0.75)
    IQR = Q3 - Q1
    mask = (num < Q1 - IQR*factor) | (num > Q3 + IQR*factor)
    return mask.sum()

def validate(df):
    '''Validates data for usability. Returns True if usable, otherwise raises an error'''
    if df.empty:
        raise ValueError("DataFrame is empty!")
    non_num = df.select_dtypes(exclude="number").columns.tolist()
    if non_num:
        raise ValueError(f"non numeric values found in  columns: {non_num}")
    if df.isna().any().any(): #returns only one True or False value if at least one value is missing in one of the columns
        raise ValueError("missing values are present!")
    return True

def clean_data(df):
    '''Returns the cleaned copy of dataset with removed whitespaces in the column names, 
    dropped duplicate rows and re-indexed rows'''
    copyy = df.copy()
    copyy.columns = [c.strip() for c in copyy.columns]
    copyy = copyy.drop_duplicates().dropna().reset_index(drop=True)
    return copyy

def quality_summary(raw, clean):
    '''Returns a plain text data quality summary'''
    lines = [f"Raw shape: {raw.shape}",
            f"Clean shape: {clean.shape}",
            f"Number of duplicate rows in Raw data: {check_duplicates(raw)}",
            "",
            "Missing values per column in Raw: ",
            check_missing(raw).to_string(),"",
            "Column types: ",
            check_types(raw).to_string(),"",
            "Outliers per column (IQR method, on clean data): ",
            find_outliers(clean).to_string(),"",
            "Descriptive statistics (on clean data): ",
            clean.describe().round(2).to_string()]
    return "\n".join(lines)

def save_clean(df, path=clean_path):
    '''Saves the cleaned data set to csv'''
    df.to_csv(path,index=False)

if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    validate(clean)
    save_clean(clean)
    summary = quality_summary(raw, clean)
    summ_path.write_text(summary)
    print(summary)
    print(f"\nSaved files: {clean_path}, {summ_path}")
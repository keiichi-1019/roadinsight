import zipfile
import pandas as pd
import io

def load_zip_csv(file):
    dfs = []
    with zipfile.ZipFile(file) as zf:
        for name in zf.namelist():
            if name.lower().endswith('.csv'):
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                    dfs.append((name, df))
    return dfs

import pandas as pd

def filter_df(df, col, filter_val, match_mode):
    vals = [v.strip() for v in filter_val.split(",") if v.strip()]
    if not vals:
        return df
    if match_mode == "完全一致":
        return df[df[col].astype(str).isin(vals)]
    else:
        pattern = "|".join(vals)
        return df[df[col].astype(str).str.contains(pattern, na=False)]

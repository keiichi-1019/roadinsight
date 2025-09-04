import streamlit as st
import pandas as pd
import zipfile
import io
import re
import unicodedata
from typing import List, Tuple, BinaryIO

st.set_page_config(page_title="ETC2.0プローブデータ加工ツール", layout="wide")

# --- 余白調整（注意文の直下とuploaderの直上だけを詰める） ---
st.markdown("""
<style>
/* 注意文ブロック自体の余白を小さくする */
.tight-after { margin-bottom: 0.1rem !important; }
.tight-after p { margin: 0.15rem 0 !important; }

/* 注意文ブロックの直後に来るFileUploaderの上余白を詰める */
.tight-after + div[data-testid="stFileUploader"] section {
  margin-top: 0.1rem !important;
  padding-top: 0.1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- ユーティリティ（正規化・型推定・条件適用） ----------
def normalize_text(s: str) -> str:
    """全角半角統一・前後空白削除"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return s.strip()

def normalize_numeric_text(s: str) -> str:
    """数値用：カンマ除去＋全角半角統一＋前後空白削除"""
    s = normalize_text(s)
    s = s.replace(",", "")
    return s

def try_parse_float(x):
    """数値に変換できればfloat、できなければNone"""
    try:
        if pd.isna(x):
            return None
        xs = normalize_numeric_text(x) if isinstance(x, str) else x
        return float(xs)
    except Exception:
        return None

def infer_column_type(series: pd.Series, sample_n: int = 100) -> str:
    """簡易推定: 'numeric' or 'string'"""
    sample = series.head(sample_n)
    ok = 0
    total = 0
    for v in sample:
        total += 1
        if try_parse_float(v) is not None:
            ok += 1
    return "numeric" if total > 0 and ok / total >= 0.9 else "string"

def apply_one_condition(df: pd.DataFrame, col: str, op: str, val: str, col_type: str) -> pd.Series:
    """
    1条件を評価してブールSeriesを返す
    op: 数値 → =, !=, >=, >, <=, <
        文字列 → =（完全一致）, !=（完全一致でない）, ~（部分一致）, !~（部分一致でない）
    """
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    s = df[col]

    if col_type == "numeric":
        left = s.apply(lambda x: try_parse_float(x))
        right = try_parse_float(val)
        if right is None:
            return pd.Series([False] * len(df), index=df.index)
        if op == "=":
            return left.eq(right)
        elif op == "!=":
            return left.ne(right)
        elif op == ">=":
            return left.ge(right)
        elif op == ">":
            return left.gt(right)
        elif op == "<=":
            return left.le(right)
        elif op == "<":
            return left.lt(right)
        else:
            return pd.Series([False] * len(df), index=df.index)
    else:
        left = s.astype(str).map(lambda x: normalize_text(x).lower())
        right = normalize_text(val).lower()
        if op == "=":         # 完全一致
            return left.eq(right)
        elif op == "!=":      # 完全一致でない
            return left.ne(right)
        elif op == "~":       # 部分一致
            return left.str.contains(re.escape(right), na=False)
        elif op == "!~":      # 部分一致でない
            return ~left.str.contains(re.escape(right), na=False)
        else:
            return pd.Series([False] * len(df), index=df.index)

def apply_conditions(df: pd.DataFrame, conditions: List[dict], type_map: dict) -> pd.DataFrame:
    """
    conditions: [{'col': 'A', 'op': '>=', 'val': '10', 'join': 'AND'}, ...]
    無効な条件（値が不正など）は除外済みを想定。
    """
    if not conditions:
        return df
    mask = None
    for idx, cond in enumerate(conditions):
        col = cond.get("col")
        op  = cond.get("op")
        val = cond.get("val", "")
        join = cond.get("join", "AND")
        col_type = type_map.get(col, "string")
        cur = apply_one_condition(df, col, op, val, col_type)
        if mask is None:
            mask = cur
        else:
            mask = (mask | cur) if join == "OR" else (mask & cur)
    return df[mask] if mask is not None else df

# --- 多重zipから最初の1つのcsvだけ返す ---
def find_first_csv_in_zip_recursive(zip_file: BinaryIO, parent_zip_name: str = ""):
    with zipfile.ZipFile(zip_file) as zf:
        for name in zf.namelist():
            if name.lower().endswith('.csv'):
                return (f"{parent_zip_name}/{name}" if parent_zip_name else name, io.BytesIO(zf.read(name)))
            elif name.lower().endswith('.zip'):
                nested_bytes = io.BytesIO(zf.read(name))
                nested_name = f"{parent_zip_name}/{name}" if parent_zip_name else name
                result = find_first_csv_in_zip_recursive(nested_bytes, nested_name)
                if result:
                    return result
    return None

# --- 多重zip内すべてのcsvを列挙（全件結合用） ---
def iter_csv_in_zip_recursive(zip_file: BinaryIO, parent_zip_name: str = "") -> List[Tuple[str, BinaryIO]]:
    result = []
    with zipfile.ZipFile(zip_file) as zf:
        for name in zf.namelist():
            if name.lower().endswith('.csv'):
                result.append((f"{parent_zip_name}/{name}" if parent_zip_name else name, io.BytesIO(zf.read(name))))
            elif name.lower().endswith('.zip'):
                nested_bytes = io.BytesIO(zf.read(name))
                nested_name = f"{parent_zip_name}/{name}" if parent_zip_name else name
                result.extend(iter_csv_in_zip_recursive(nested_bytes, nested_name))
    return result

# --- Excel風の列名生成 ---
def get_excel_column_name(idx):
    result = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = chr(65 + rem) + result
    return result

def assign_excel_columns_to_df(df, excel_cols):
    df = df.copy()
    df.columns = excel_cols
    return df

# --- プレビュー用：最初の1ファイル・1csvのみ ---
def load_preview_df(uploaded_files):
    if not uploaded_files:
        return None, None, None
    upfile = uploaded_files[0]
    try:
        if upfile.name.endswith('.zip'):
            result = find_first_csv_in_zip_recursive(upfile, upfile.name)
            if result:
                file_disp_name, cf = result
                df = pd.read_csv(cf, header=0)
                col_count = len(df.columns)
                excel_cols = [get_excel_column_name(i) for i in range(col_count)]
                df = assign_excel_columns_to_df(df, excel_cols)
                return df, excel_cols, file_disp_name
        else:
            df = pd.read_csv(upfile, header=0)
            col_count = len(df.columns)
            excel_cols = [get_excel_column_name(i) for i in range(col_count)]
            df = assign_excel_columns_to_df(df, excel_cols)
            return df, excel_cols, upfile.name
    except Exception as e:
        st.error(f"{upfile.name}の読み込みでエラー: {e}")
    return None, None, None

def get_file_row_counts(uploaded_files):
    file_counts = []
    total = 0
    for upfile in uploaded_files:
        try:
            if upfile.name.endswith('.zip'):
                csv_files = iter_csv_in_zip_recursive(upfile, upfile.name)
                for file_disp_name, cf in csv_files:
                    df = pd.read_csv(cf, header=0)
                    file_counts.append((file_disp_name, len(df)))
                    total += len(df)
            else:
                df = pd.read_csv(upfile, header=0)
                file_counts.append((upfile.name, len(df)))
                total += len(df)
        except Exception as e:
            file_counts.append((upfile.name, f"エラー: {e}"))
    return file_counts, total

# ------------------------------- UI本体 -------------------------------
def show_common_tab():
    st.header("データ結合・抽出")

    # --- 注意文（このブロックの直後にuploaderが来る前提） ---
    st.markdown("""
    <div class="tight-after">
      <p>データを結合、抽出することができます</p>
      <p>ZIP/CSVファイルをアップロードしてください（複数選択可能）</p>
      <p>※各ファイル最大30GBまで対応（ただし容量の大きなファイルは処理できない可能性があります）</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "",
        type=['zip', 'csv'],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.info("ファイルをアップロードしてください。")
        return

    preview_df, excel_cols, preview_name = load_preview_df(uploaded_files)
    if preview_df is None:
        st.warning("CSVデータが正しく読み込めませんでした。")
        return

    # 列タイプの推定
    inferred_types = {c: infer_column_type(preview_df[c]) for c in preview_df.columns}

    disp = preview_df.copy()
    disp.index = range(1, len(disp) + 1)
    st.subheader("▼ サンプルデータ（先頭10行）")
    st.caption(f"プレビュー対象: {preview_name}")
    st.dataframe(disp.head(10))

    st.subheader("抽出する列を選択してください（順序も反映）")
    if 'column_selector' not in st.session_state or st.session_state.column_selector is None:
        st.session_state.column_selector = excel_cols
    selected_cols = st.multiselect(
        "列の選択（表示順・抽出順がそのまま反映されます）",
        options=excel_cols,
        default=st.session_state.column_selector,
        key='column_selector'
    )

    # ---------- 新：条件ビルダー ----------
    st.subheader("▼ 列ごとに値でフィルタ（値は1個、条件＆AND/ORを指定）")

    available_cols = selected_cols or excel_cols

    # セッション初期化
    if 'conditions' not in st.session_state:
        st.session_state.conditions = [
            {"join": "—", "col": available_cols[0] if available_cols else "", "op": "=", "val": ""}
        ]

    # 演算子（表示ラベル ↔ コード）
    OPS_NUM = [("=", "=（等しい）"), ("!=", "!=（等しくない）"),
               (">=", ">=（以上）"), (">", ">（より大きい）"),
               ("<=", "<=（以下）"), ("<", "<（より小さい）")]
    OPS_STR = [("=", "=（完全一致）"), ("!=", "!=（完全一致でない）"),
               ("~", "〜（部分一致）"), ("!~", "!〜（部分一致でない）")]

    # 条件行の描画
    for i, cond in enumerate(st.session_state.conditions):
        st.markdown(f"**条件 {i+1}**")
        # 行レイアウト：先頭行は連結なし
        if i == 0:
            cols = st.columns([2, 2, 3, 1])  # 列, 演算子, 値, 削除
        else:
            cols = st.columns([1, 2, 2, 3, 1])  # 連結, 列, 演算子, 値, 削除

        # 連結（2行目以降）
        if i > 0:
            join_default = "AND" if cond.get("join", "AND") not in ["AND", "OR"] else cond.get("join", "AND")
            join = cols[0].selectbox("連結", options=["AND", "OR"],
                                     index=0 if join_default == "AND" else 1,
                                     key=f"join_{i}")
        else:
            join = "—"  # 内部的に保持

        # 列選択
        cur_col = cond.get("col", available_cols[0] if available_cols else "")
        col_sel = cols[0 if i == 0 else 1].selectbox("列", options=available_cols,
                            index=(available_cols.index(cur_col) if cur_col in available_cols else 0),
                            key=f"col_{i}")
        # 型推定表示
        (cols[0 if i == 0 else 1]).caption(f"推定: {'数値' if inferred_types.get(col_sel,'string')=='numeric' else '文字列'}")

        # 演算子
        col_type = inferred_types.get(col_sel, "string")
        ops = OPS_NUM if col_type == "numeric" else OPS_STR
        # 現在コード→表示index
        cur_op_code = cond.get("op", ops[0][0])
        op_codes = [c for c, _ in ops]
        if cur_op_code not in op_codes:
            cur_op_code = ops[0][0]
        op_label_idx = op_codes.index(cur_op_code)
        op_label = cols[1 if i == 0 else 2].selectbox("条件（演算子）",
                            options=[lab for _, lab in ops],
                            index=op_label_idx,
                            key=f"op_{i}")
        # 表示ラベル→コード
        selected_code = ops[[lab for _, lab in ops].index(op_label)][0]

        # 値（1個）＋バリデーション
        cur_val = cond.get("val", "")
        val_in = cols[2 if i == 0 else 3].text_input("値", value=cur_val, key=f"val_{i}")
        invalid = False
        if ("," in val_in) or ("、" in val_in):
            cols[2 if i == 0 else 3].error("値は1個のみ入力してください（カンマ区切りは不可）")
            invalid = True
        if normalize_text(val_in) == "":
            cols[2 if i == 0 else 3].error("値を入力してください")
            invalid = True

        # 削除ボタン（1クリックで即削除）
        if cols[3 if i == 0 else 4].button("－削除", key=f"rm_{i}"):
            # 先頭行も削除可。空になったら新規1行を作る
            st.session_state.conditions.pop(i)
            if not st.session_state.conditions:
                st.session_state.conditions.append({"join": "—", "col": available_cols[0] if available_cols else "", "op": "=", "val": ""})
            st.rerun()

        # 値が有効なら状態更新
        st.session_state.conditions[i] = {"join": join, "col": col_sel, "op": selected_code, "val": val_in, "invalid": invalid}

        st.divider()

    c1, c2 = st.columns([1, 3])
    if c1.button("＋ 条件を追加"):
        st.session_state.conditions.append({
            "join": "AND" if len(st.session_state.conditions) > 0 else "—",
            "col": available_cols[0] if available_cols else "",
            "op": "=",
            "val": "",
            "invalid": True
        })
        st.rerun()

    # --------- 抽出プレビュー（条件適用） ---------
    if 'merged_preview' not in st.session_state:
        st.session_state.merged_preview = None
        st.session_state.merged_preview_nrows = 0

    if st.button('この条件で抽出する'):
        preview = preview_df.copy()
        if selected_cols:
            preview = preview[selected_cols]
        # 無効行を除外して適用
        valid_conds = [ {"join": c["join"], "col": c["col"], "op": c["op"], "val": c["val"]}
                        for c in st.session_state.conditions if not c.get("invalid", False) ]
        if valid_conds:
            preview = apply_conditions(preview, valid_conds, inferred_types)
        preview.index = range(1, len(preview) + 1)
        st.session_state.merged_preview = preview
        st.session_state.merged_preview_nrows = len(preview_df if not selected_cols else preview_df[selected_cols])

    if st.session_state.merged_preview is not None:
        n_all = st.session_state.merged_preview_nrows
        n_hit = len(st.session_state.merged_preview)
        if n_all > 0:
            n_bar = round(100 * n_hit / n_all)
            if n_bar == 0 and n_hit > 0:
                n_bar = 1
        else:
            n_bar = 0
        if n_hit == 0:
            st.info("1/100以下の行数に抽出")
        else:
            st.info(f"約{n_bar}/100の行数に抽出")
        st.subheader("▼ 抽出したサンプル（先頭10行）")
        st.dataframe(st.session_state.merged_preview.head(10))

    st.markdown("---")
    st.subheader("抽出して結合・ダウンロードする")

    if 'csv_bytes' not in st.session_state:
        st.session_state.csv_bytes = None
        st.session_state.last_csv_params = (None, None)

    do_process = st.button('抽出して結合・ダウンロードする')

    cur_params = (
        tuple(selected_cols) if selected_cols else (),
        tuple((c["join"], c["col"], c["op"], c["val"]) for c in st.session_state.conditions if not c.get("invalid", False))
    )
    if cur_params != st.session_state.last_csv_params:
        st.session_state.csv_bytes = None
        st.session_state.last_csv_params = cur_params

    if do_process or (st.session_state.csv_bytes is not None):
        if st.session_state.csv_bytes is None:
            progress = st.progress(0, text='抽出処理中...')
            dfs = []
            n_files = len(uploaded_files)
            for i, upfile in enumerate(uploaded_files):
                try:
                    if upfile.name.endswith('.zip'):
                        csv_files = iter_csv_in_zip_recursive(upfile, upfile.name)
                        for file_disp_name, cf in csv_files:
                            df = pd.read_csv(cf, header=0)
                            col_count = len(df.columns)
                            excel_cols_cur = [get_excel_column_name(j) for j in range(col_count)]
                            df = assign_excel_columns_to_df(df, excel_cols_cur)
                            inferred_types_cur = {c: infer_column_type(df[c]) for c in df.columns}
                            if selected_cols:
                                for add_idx in range(len(selected_cols) - len(df.columns)):
                                    df[f"dummy_{add_idx}"] = ''
                                df = df[selected_cols]
                            valid_conds = [ {"join": c["join"], "col": c["col"], "op": c["op"], "val": c["val"]}
                                            for c in st.session_state.conditions if not c.get("invalid", False) ]
                            if valid_conds:
                                df = apply_conditions(df, valid_conds, inferred_types_cur)
                            dfs.append(df)
                    else:
                        df = pd.read_csv(upfile, header=0)
                        col_count = len(df.columns)
                        excel_cols_cur = [get_excel_column_name(j) for j in range(col_count)]
                        df = assign_excel_columns_to_df(df, excel_cols_cur)
                        inferred_types_cur = {c: infer_column_type(df[c]) for c in df.columns}
                        if selected_cols:
                            for add_idx in range(len(selected_cols) - len(df.columns)):
                                df[f"dummy_{add_idx}"] = ''
                            df = df[selected_cols]
                        valid_conds = [ {"join": c["join"], "col": c["col"], "op": c["op"], "val": c["val"]}
                                        for c in st.session_state.conditions if not c.get("invalid", False) ]
                        if valid_conds:
                            df = apply_conditions(df, valid_conds, inferred_types_cur)
                        dfs.append(df)
                except Exception as e:
                    st.error(f"{upfile.name}の読み込みでエラー: {e}")
                progress.progress((i+1)/(n_files+1), text=f"{i+1}/{n_files} ファイル抽出・結合処理中...")
            progress.progress(n_files/(n_files+1), text="抽出・CSV生成中...")
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                merged.index = range(1, len(merged) + 1)
                st.session_state.csv_bytes = merged.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                file_counts, _ = get_file_row_counts(uploaded_files)
                st.session_state.merged_preview = merged
                st.session_state.merged_preview_nrows = sum([cnt if isinstance(cnt, int) else 0 for _, cnt in file_counts])
                progress.progress(1.0, text="完了")
            else:
                progress.empty()
                st.warning("結合できるデータがありませんでした。")
        if st.session_state.csv_bytes is not None:
            st.download_button(
                'ダウンロード開始（merged_filtered_data.csv）',
                data=st.session_state.csv_bytes,
                file_name='merged_filtered_data.csv',
                mime='text/csv'
            )

def main():
    show_common_tab()

if __name__ == "__main__":
    main()

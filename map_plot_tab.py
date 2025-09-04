# map_plot_tab.py
# -----------------------------------------------------------------------------
# アップロード耐性を強化（チャンク読み＋配列ベースのサンプリング）
# ※描画は Folium の元実装に準拠（PyDeck等は使いません）
# -----------------------------------------------------------------------------

from __future__ import annotations
import io
import os
import json
import math
import zipfile
import tempfile
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- optional orjson（あれば高速） ------------------------------------------
try:
    import orjson
    def _dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def _dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

# --- 画像・定数 --------------------------------------------------------------
EXAMPLE_DIR = "assets/map_examples"
PLOT_IMG = f"{EXAMPLE_DIR}/Plot_display_example.png"
MESH_IMG = f"{EXAMPLE_DIR}/Mesh_display_example.png"
HEAT_IMG = f"{EXAMPLE_DIR}/Heat_map_display_example.png"

# アップロード～前処理の安全側パラメータ
MAX_POINTS   = 150_000        # 個点表示向けに保持する最大件数（ここを超えないようサンプル化）
CHUNK_ROWS   = 100_000        # CSV/ZIP 読み込み時のチャンク行数
RANDOM_SEED  = 0              # サンプリング乱数シード

# ---------- 小物ユーティリティ ----------------------------------------------
def _idx_to_excel_col(idx: int) -> str:
    s, n = "", idx
    while True:
        n, r = divmod(n, 26)
        s = chr(65 + r) + s
        if n == 0:
            break
        n -= 1
    return s

def _excel_label_map(cols: Iterable[str]):
    """{'A': colname0, 'B': colname1, ...} を返す"""
    return {_idx_to_excel_col(i): c for i, c in enumerate(cols)}

def _score_in_range(series: pd.Series, low: float, high: float) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    if len(v) < 10:
        return -1.0
    in_range = v.between(low, high).mean()
    frac = (np.abs(v - np.round(v)) > 1e-6).mean()
    return in_range * 0.85 + frac * 0.15

def _autodetect_lon_lat(df: pd.DataFrame):
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        return None, None
    lon = max(nums, key=lambda c: _score_in_range(df[c], 122, 154))
    lat = max(nums, key=lambda c: _score_in_range(df[c], 20, 46))
    if lon == lat:
        alts = sorted(nums, key=lambda c: _score_in_range(df[c], 20, 46), reverse=True)
        for a in alts:
            if a != lon:
                lat = a
                break
    if _score_in_range(df[lon], 122, 154) < 0.7:
        lon = None
    if _score_in_range(df[lat], 20, 46) < 0.7:
        lat = None
    return lon, lat

# ---------- アップロード：チャンク読みの下ごしらえ --------------------------
def _reset_uploaded_file(file):
    try:
        file.seek(0)
    except Exception:
        pass

def _open_first_csv_in_zip(file) -> io.TextIOBase:
    zf = zipfile.ZipFile(file)
    for n in zf.namelist():
        if n.lower().endswith(".csv"):
            return io.TextIOWrapper(zf.open(n, "r"), encoding="utf-8", errors="ignore")
    raise ValueError("ZIP内にCSVが見つかりません。")

def _iter_csv_chunks(uploaded_file, usecols=None, dtype=None):
    """
    Streamlit UploadedFile（CSV/ZIP）から pandas チャンクを順次返す
    """
    _reset_uploaded_file(uploaded_file)
    name = (uploaded_file.name or "").lower()
    if name.endswith(".zip"):
        f = _open_first_csv_in_zip(uploaded_file)
        yield from pd.read_csv(f, chunksize=CHUNK_ROWS, usecols=usecols, dtype=dtype)
    else:
        yield from pd.read_csv(uploaded_file, chunksize=CHUNK_ROWS, usecols=usecols, dtype=dtype)

def _autodetect_with_sample(uploaded_file, sample_rows=50_000):
    """
    自動判定は先頭数万行だけで実施（重い全読込をしない）
    """
    it = _iter_csv_chunks(uploaded_file)
    head = next(it)
    if len(head) > sample_rows:
        head = head.head(sample_rows)
    lon, lat = _autodetect_lon_lat(head)
    return lon, lat, head

# ---------- ベクトル化したサンプル抽出（アップロード後の主処理） ------------
def _stream_sample_points(uploaded_file, value_col: str | None, lon_col: str, lat_col: str, keep_pct=100.0):
    """
    逐次読み＋確率間引き（ベクトル化）＋配列ベースのリザーバで MAX_POINTS 以内に収める。
    戻り値: DataFrame(lat, lon[, val]) すべて float32
    """
    p = max(0.0001, min(1.0, float(keep_pct) / 100.0))
    rng = np.random.default_rng(RANDOM_SEED)
    usecols = [c for c in [lat_col, lon_col, value_col] if c]

    lat_keep = np.empty((0,), dtype="float32")
    lon_keep = np.empty((0,), dtype="float32")
    if value_col:
        val_keep = np.empty((0,), dtype="float32")
    else:
        val_keep = None

    for chunk in _iter_csv_chunks(uploaded_file, usecols=usecols):
        w = chunk.rename(columns={lat_col: "lat", lon_col: "lon"})
        w["lat"] = pd.to_numeric(w["lat"], errors="coerce")
        w["lon"] = pd.to_numeric(w["lon"], errors="coerce")
        if value_col:
            w["val"] = pd.to_numeric(w[value_col], errors="coerce")
        w = w.dropna(subset=["lat", "lon"])
        w = w[w["lat"].between(20, 46) & w["lon"].between(122, 154)]
        if w.empty:
            continue

        lat = w["lat"].to_numpy(dtype="float32", copy=False)
        lon = w["lon"].to_numpy(dtype="float32", copy=False)
        lat = np.round(lat, 5); lon = np.round(lon, 5)
        if value_col:
            val = w["val"].to_numpy(dtype="float32", copy=False)

        # 確率間引き（完全ベクトル化）
        if p < 1.0:
            keep_mask = rng.random(lat.shape[0]) < p
            if not keep_mask.any():
                continue
            lat = lat[keep_mask]; lon = lon[keep_mask]
            if value_col: val = val[keep_mask]

        # 連結
        if value_col:
            lat_cat = np.concatenate([lat_keep, lat], dtype="float32")
            lon_cat = np.concatenate([lon_keep, lon], dtype="float32")
            val_cat = np.concatenate([val_keep, val], dtype="float32")
        else:
            lat_cat = np.concatenate([lat_keep, lat], dtype="float32")
            lon_cat = np.concatenate([lon_keep, lon], dtype="float32")

        # 上限を超えるときはランダム抽出で MAX_POINTS へ
        n = lat_cat.shape[0]
        if n > MAX_POINTS:
            idx = rng.choice(n, size=MAX_POINTS, replace=False)
            idx.sort()
            lat_keep = lat_cat[idx]
            lon_keep = lon_cat[idx]
            if value_col:
                val_keep = val_cat[idx]
        else:
            lat_keep = lat_cat
            lon_keep = lon_cat
            if value_col:
                val_keep = val_cat

    if lat_keep.size == 0:
        return pd.DataFrame(columns=["lat", "lon"] + (["val"] if value_col else []))

    if value_col:
        return pd.DataFrame({"lat": lat_keep, "lon": lon_keep, "val": val_keep})
    else:
        return pd.DataFrame({"lat": lat_keep, "lon": lon_keep})

# ---------- （任意）分割保存：Parquet/CSV に吐きながら処理 -------------------
def _split_save(uploaded_file, lat_col, lon_col, value_col=None, rows_per_part=1_000_000, use_parquet=True):
    """
    希望があれば、アップロード後に取り込んだデータを rows_per_part 行ごとに一時フォルダへ分割保存
    （描画には使わない。オフライン処理や後続バッチ向け）
    """
    tmpdir = tempfile.mkdtemp(prefix="split_ingest_")
    part_idx = 0
    buf_rows = 0
    lat_buf = np.empty((0,), dtype="float32")
    lon_buf = np.empty((0,), dtype="float32")
    val_buf = np.empty((0,), dtype="float32") if value_col else None

    for chunk in _iter_csv_chunks(uploaded_file, usecols=[lat_col, lon_col, value_col] if value_col else [lat_col, lon_col]):
        w = chunk.rename(columns={lat_col: "lat", lon_col: "lon"})
        w["lat"] = pd.to_numeric(w["lat"], errors="coerce")
        w["lon"] = pd.to_numeric(w["lon"], errors="coerce")
        if value_col:
            w["val"] = pd.to_numeric(w[value_col], errors="coerce")
        w = w.dropna(subset=["lat", "lon"])
        w = w[w["lat"].between(20, 46) & w["lon"].between(122, 154)]
        if w.empty:
            continue

        lat = w["lat"].to_numpy("float32"); lon = w["lon"].to_numpy("float32")
        lat = np.round(lat, 5); lon = np.round(lon, 5)
        if value_col:
            val = w["val"].to_numpy("float32")

        # 連結
        if value_col:
            lat_cat = np.concatenate([lat_buf, lat], dtype="float32")
            lon_cat = np.concatenate([lon_buf, lon], dtype="float32")
            val_cat = np.concatenate([val_buf, val], dtype="float32")
        else:
            lat_cat = np.concatenate([lat_buf, lat], dtype="float32")
            lon_cat = np.concatenate([lon_buf, lon], dtype="float32")

        start = 0
        n = lat_cat.shape[0]
        while (n - start) >= rows_per_part:
            end = start + rows_per_part
            dfp = pd.DataFrame({
                "lat": lat_cat[start:end],
                "lon": lon_cat[start:end],
                **({"val": val_cat[start:end]} if value_col else {})
            })
            part_path = os.path.join(tmpdir, f"part_{part_idx:05d}" + (".parquet" if use_parquet else ".csv"))
            if use_parquet:
                try:
                    dfp.to_parquet(part_path, index=False)
                except Exception:
                    dfp.to_csv(part_path, index=False)
            else:
                dfp.to_csv(part_path, index=False)
            part_idx += 1
            start = end

        # 余りをバッファに残す
        lat_buf = lat_cat[start:]
        lon_buf = lon_cat[start:]
        if value_col:
            val_buf = val_cat[start:]

    # 最後の端数も出力
    if lat_buf.size:
        dfp = pd.DataFrame({
            "lat": lat_buf,
            "lon": lon_buf,
            **({"val": val_buf} if value_col else {})
        })
        part_path = os.path.join(tmpdir, f"part_{part_idx:05d}" + (".parquet" if use_parquet else ".csv"))
        try:
            if use_parquet:
                dfp.to_parquet(part_path, index=False)
            else:
                dfp.to_csv(part_path, index=False)
        except Exception:
            dfp.to_csv(part_path, index=False)
        part_idx += 1

    return tmpdir, part_idx

# ---------- パレット＆レンジ --------------------------------------------------
BASE_PALETTES = {
    "Red→Blue": ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"],  # 小→大で赤→青
    "Blue→Red": ["#4575b4", "#91bfdb", "#fee090", "#fc8d59", "#d73027"],
    "Viridis":  ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151"],
    "Plasma":   ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636"],
    "Cividis":  ["#00204d", "#2b4066", "#566979", "#8f9c7f", "#d4e06b"],
    "Greys":    ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"],
}
def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0,2,4))
def _rgb_to_hex(rgb): return "#%02x%02x%02x" % rgb
def _lerp(a,b,t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))
def _palette_to_bins(base_colors, bins: int):
    if bins <= 1: return [base_colors[0]]
    pts = np.linspace(0, len(base_colors) - 1, bins)
    out = []
    for p in pts:
        i = int(np.floor(p))
        if i >= len(base_colors) - 1:
            out.append(base_colors[-1]); continue
        t = p - i
        a = _hex_to_rgb(base_colors[i]); b = _hex_to_rgb(base_colors[i+1])
        out.append(_rgb_to_hex(_lerp(a, b, t)))
    return out
def _build_breaks_auto(vmin: float, vmax: float, bins: int, inner_bounds: List[float] | None):
    if inner_bounds and len(inner_bounds) == bins - 1:
        bounds = sorted(float(x) for x in inner_bounds)
    else:
        bounds = list(np.linspace(vmin, vmax, bins + 1)[1:-1])
    return [float(vmin)] + bounds + [float(vmax)]

# ---------- Folium 用（個点） -------------------------------------------------
def _features_with_color(work: pd.DataFrame, breaks, colors, progress):
    import bisect
    coords = np.column_stack((work["lon"].to_numpy(), work["lat"].to_numpy()))
    has_val = "val" in work.columns
    vals = work["val"].to_numpy() if has_val else None

    feats = []; n = len(coords); step = max(1, n // 80)
    for i, (x, y) in enumerate(coords, 1):
        if has_val:
            j = bisect.bisect_right(breaks, float(vals[i-1])) - 1
            j = max(0, min(j, len(colors) - 1))
            col = colors[j]
        else:
            col = "#E67814"
        feats.append({
            "type": "Feature",
            "properties": {"c": col},
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
        })
        if i % step == 0 or i == n:
            progress.progress(10 + int(80 * i / n), text=f"点データ準備中... {i:,}/{n:,}")
    return feats

# ---------- メッシュ（Folium） -----------------------------------------------
def _estimate_deg_for_km(cell_km: float, center_lat: float):
    deg_lat = cell_km / 111.0
    cosphi = max(0.1, math.cos(math.radians(center_lat)))
    deg_lon = cell_km / (111.0 * cosphi)
    return deg_lon, deg_lat
def _make_gradient(min_hex: str, max_hex: str, bins: int):
    a = _hex_to_rgb(min_hex); b = _hex_to_rgb(max_hex)
    cols = []
    for i in range(bins):
        t = i / (bins - 1) if bins > 1 else 0.0
        cols.append(_rgb_to_hex(_lerp(a, b, t)))
    return cols
def _mesh_geojson_by_cellsize(work: pd.DataFrame, cell_km: float, colors, bins=7):
    min_lon, max_lon = float(work["lon"].min()), float(work["lon"].max())
    min_lat, max_lat = float(work["lat"].min()), float(work["lat"].max())
    lat0 = float(work["lat"].mean())
    dlon, dlat = _estimate_deg_for_km(cell_km, lat0)

    pad_lon = (max_lon - min_lon) * 0.01 or dlon
    pad_lat = (max_lat - min_lat) * 0.01 or dlat
    min_lon -= pad_lon; max_lon += pad_lon
    min_lat -= pad_lat; max_lat += pad_lat

    nx = max(1, int(math.ceil((max_lon - min_lon) / dlon)))
    ny = max(1, int(math.ceil((max_lat - min_lat) / dlat)))
    nx = min(nx, 800); ny = min(ny, 800)

    H, xedges, yedges = np.histogram2d(
        work["lon"].to_numpy(), work["lat"].to_numpy(),
        bins=[nx, ny], range=[[min_lon, max_lon], [min_lat, max_lat]]
    )
    counts = H.T

    qs = np.linspace(0, 1, bins + 1)[1:-1]
    thr = [float(np.quantile(counts[counts > 0], q)) if np.any(counts > 0) else 1.0 for q in qs]

    def _color_for(v):
        if v <= 0:
            return None
        for i, t in enumerate(thr):
            if v <= t:
                return colors[i]
        return colors[-1]

    feats = []
    for iy in range(ny):
        for ix in range(nx):
            v = counts[iy, ix]
            col = _color_for(v)
            if col is None:
                continue
            x0, x1 = xedges[ix], xedges[ix+1]
            y0, y1 = yedges[iy], yedges[i+1]
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            feats.append({
                "type": "Feature",
                "properties": {"c": col, "count": float(v)},
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            })
    return {"type": "FeatureCollection", "features": feats}, thr

# ---------- 凡例 -------------------------------------------------------------
def _legend_block(items):
    html = '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;">'
    for label, col in items:
        html += (
            f'<span style="display:inline-flex;align-items:center;gap:8px;">'
            f'<span style="display:inline-block;width:18px;height:12px;background:{col};'
            f'border:1px solid #888;"></span>'
            f'<span style="font-size:12px;">{label}</span>'
            f'</span>'
        )
    html += "</div>"
    return html
def _legend_for_points(breaks, colors):
    items = []
    for i in range(len(colors)):
        lo = breaks[i]; hi = breaks[i+1]
        if i == 0: label = f"[{lo:.2g}, {hi:.2g}]"
        else:      label = f"({lo:.2g}, {hi:.2g}]"
        items.append((label, colors[i]))
    return _legend_block(items)
def _legend_for_mesh(thr, colors):
    bounds = [0.0] + [float(x) for x in thr]
    items = []
    for i, col in enumerate(colors):
        lo = bounds[i]; hi = bounds[i+1] if i < len(thr) else "max"
        if hi == "max": label = f"({lo:.2g}, max]"
        elif i == 0:    label = f"[>0, {hi:.2g}]"
        else:           label = f"({lo:.2g}, {hi:.2g}]"
        items.append((label, col))
    return _legend_block(items)
def _legend_for_heatmap():
    return _legend_block([("低密度", "#ffffb2"), ("中密度", "#fd8d3c"), ("高密度", "#bd0026")])

# ---------- メインタブ -------------------------------------------------------
def show_map_plot_tab():
    st.header("地図にプロット")

    c1, c2 = st.columns([1, 0.25])
    with c1:
        st.write("緯度経度を持つデータを地図にプロットできます")
    with c2:
        with st.popover("表示例を見る（クリック）"):
            st.caption("プロット・メッシュ・ヒートマップの表示例")
            tabs = st.tabs(["個々の点をプロット", "メッシュ表示", "ヒートマップ"])
            with tabs[0]: st.image(PLOT_IMG, use_container_width=True)
            with tabs[1]: st.image(MESH_IMG, use_container_width=True)
            with tabs[2]: st.image(HEAT_IMG, use_container_width=True)

    st.write("ZIP/CSVファイルをアップロードしてください（複数選択できません）")
    st.write("※大きなファイルはサンプリングして取り込みます。必要に応じて分割保存も可能です。")

    up = st.file_uploader("", type=["zip", "csv"], accept_multiple_files=False, label_visibility="collapsed")
    if not up:
        st.info("ファイルをアップロードしてください。")
        return

    # 入口ガード（任意）：超大サイズは注意喚起
    size_mb = getattr(up, "size", 0) / (1024 * 1024)
    if size_mb and size_mb > 150:
        st.warning("このサイズだとブラウザ→サーバ転送で失敗する可能性があります。"
                   "それでも続ける場合は取り込み後に強制サンプリングします。")

    # --- プレビュー＆自動判定（先頭のみ） -----------------------------------
    try:
        auto_lon, auto_lat, head_df = _autodetect_with_sample(up)
    except Exception as e:
        st.exception(e); return
    if head_df is None or head_df.empty:
        st.warning("有効なデータが見つかりません"); return

    st.caption(f"プレビュー: {up.name}")
    label_map = _excel_label_map(head_df.columns)
    preview = head_df.head(10).copy()
    preview.columns = list(label_map.keys())

    lat_label_auto = next((k for k, v in label_map.items() if v == auto_lat), None)
    lon_label_auto = next((k for k, v in label_map.items() if v == auto_lon), None)

    def _style_preview(sdf: pd.DataFrame):
        styles = pd.DataFrame("", index=sdf.index, columns=sdf.columns)
        if lat_label_auto in sdf.columns: styles[lat_label_auto] = "background-color: #FFF7CC"  # 緯度＝淡黄
        if lon_label_auto in sdf.columns: styles[lon_label_auto] = "background-color: #CCE5FF"  # 経度＝淡青
        return styles
    st.dataframe(preview.style.apply(_style_preview, axis=None))

    # ---- 緯度/経度 手動指定（列ラベルから選択） ----
    with st.expander("緯度・経度の列が自動判定されない場合はこちらを設定",
                     expanded=(auto_lon is None or auto_lat is None)):
        labels = list(label_map.keys())
        c3, c4 = st.columns(2)
        lat_label_sel = c3.selectbox(
            "緯度の列を指定してください",
            options=["自動検出"] + labels,
            index=(labels.index(lat_label_auto) + 1) if lat_label_auto in labels else 0
        )
        lon_label_sel = c4.selectbox(
            "経度の列を指定してください",
            options=["自動検出"] + labels,
            index=(labels.index(lon_label_auto) + 1) if lon_label_auto in labels else 0
        )
        lat_override = None if lat_label_sel == "自動検出" else label_map[lat_label_sel]
        lon_override = None if lon_label_sel == "自動検出" else label_map[lon_label_sel]

    # ---- 描画オプション -----------------------------------------------------
    st.subheader("描画オプション")
    use_photo = st.checkbox("ベースマップを航空写真にする", value=False)

    samp_labels = ["1/10,000（0.01%）", "1/1,000（0.1%）", "1/100（1%）", "1/10（10%）", "1/1（100%）"]
    samp_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    copt1, copt2 = st.columns([1, 1.4])
    samp_label = copt1.select_slider("サンプリング（％）", options=samp_labels, value=samp_labels[-1],
                                     help="抽出割合を5段階から選択")
    sampling_pct = samp_values[samp_labels.index(samp_label)]

    draw_method = copt2.selectbox(
        "描画方法",
        options=["個々の点をプロット", "メッシュ表示", "ヒートマップ"],
        index=0,
        help="データ量に応じて選択してください"
    )

    # 値で色分け（個々の点のみ）
    use_color = False
    bins = 5
    base_palette = "Red→Blue"
    point_radius = 2.2
    value_col = None
    if draw_method == "個々の点をプロット":
        st.markdown("**値で色分け**")
        use_color = st.checkbox("値で色分けする", value=False)
        point_radius = st.slider("点のサイズ（半径・px）", 1.0, 8.0, 2.2, 0.2)
        if use_color:
            labels = list(label_map.keys())
            value_label = st.selectbox("色分けに使う列ラベルを選択してください", options=labels, index=0)
            value_col = label_map[value_label]
            s = pd.to_numeric(head_df[value_col], errors="coerce")
            vmin_auto = float(s.min(skipna=True)); vmax_auto = float(s.max(skipna=True))
            c5, c6 = st.columns([1, 2])
            bins = int(c5.number_input("レンジ数", min_value=3, max_value=9, value=5, step=1))
            base_palette = c6.selectbox("使用するカラーセット", list(BASE_PALETTES.keys()), index=0)

            with st.expander("カラーや閾値の詳細設定", expanded=False):
                init_bounds = list(np.linspace(vmin_auto, vmax_auto, bins + 1)[1:-1])
                init_colors = _palette_to_bins(BASE_PALETTES[base_palette], bins)
                inner_bounds_ui, colors_ui = [], []
                for i in range(bins):
                    cc1, cc2 = st.columns([1, 1])
                    if i < bins - 1:
                        b = cc1.number_input(f"閾値{i+1}", value=float(init_bounds[i]), key=f"thr_{i}")
                        inner_bounds_ui.append(b)
                    else:
                        cc1.markdown("&nbsp;", unsafe_allow_html=True)
                    col = cc2.color_picker(f"カラー{i+1}", value=init_colors[i], key=f"color_{i}")
                    colors_ui.append(col)
                st.session_state["ui_inner_bounds"] = inner_bounds_ui
                st.session_state["ui_colors"] = colors_ui
                st.session_state["ui_vmin_auto"] = vmin_auto
                st.session_state["ui_vmax_auto"] = vmax_auto
                st.session_state["color_value_label"] = value_label
        else:
            st.session_state["ui_inner_bounds"] = None
            st.session_state["ui_colors"] = None
            st.session_state["color_value_label"] = None
    else:
        st.session_state["ui_inner_bounds"] = None
        st.session_state["ui_colors"] = None
        st.session_state["color_value_label"] = None

    # メッシュ設定
    mesh_cell_km = None
    mesh_min_color = "#ffff00"
    mesh_max_color = "#d73027"
    if draw_method == "メッシュ表示":
        size_labels = [
            "10 km", "5 km", "2 km", "1 km", "0.5 km",
            "0.25 km", "0.2 km", "0.15 km", "0.125 km", "0.1 km（100 m）"
        ]
        size_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.2, 0.15, 0.125, 0.1]
        m1, m2 = st.columns([1, 1])
        sel = m1.select_slider("メッシュサイズ", options=size_labels, value=size_labels[3])
        mesh_cell_km = size_values[size_labels.index(sel)]
        mesh_min_color = m2.color_picker("メッシュ最小レンジ色", value=mesh_min_color)
        mesh_max_color = st.color_picker("メッシュ最大レンジ色", value=mesh_max_color)

    # ヒートマップ設定
    heat_level = None
    if draw_method == "ヒートマップ":
        level_label = st.select_slider(
            "密度の見せ方",
            options=["細かい", "やや細かい", "普通", "やや粗い", "粗い"],
            value="普通",
        )
        lvl_map = {"細かい": (6, 8), "やや細かい": (10, 12), "普通": (15, 18), "やや粗い": (20, 24), "粗い": (30, 36)}
        heat_level = lvl_map[level_label]

    # 分割保存（任意）
    with st.expander("（任意）アップロード後に分割保存したい場合はこちら"):
        enable_split = st.checkbox("取り込み時に分割保存する（オフのままでOK）", value=False)
        rows_per_part = st.number_input("1パートの行数", 100_000, 5_000_000, 1_000_000, 100_000)
        use_parquet = st.checkbox("Parquetで保存（高速・高圧縮。×でCSV）", value=True)

    # ---- UI配置（順序固定） -------------------------------------------------
    start = st.button("地図を描画する")
    progress_slot = st.empty()
    map_slot = st.empty()
    download_slot = st.empty()
    legend_slot = st.empty()
    if not start:
        return

    progress = progress_slot.progress(0, text="前処理中...")

    # 値色分け列（個点のみ）
    if draw_method == "個々の点をプロット" and use_color and st.session_state.get("color_value_label"):
        value_col = label_map[st.session_state["color_value_label"]]
    else:
        value_col = None

    # 緯度経度列の確定
    lon_col = lon_override if lon_override else (auto_lon or None)
    lat_col = lat_override if lat_override else (auto_lat or None)
    if not lon_col or not lat_col:
        progress.progress(100, text="エラー"); st.error("緯度・経度列を決定できません"); return

    # （任意）分割保存
    if enable_split:
        tmpdir, n_parts = _split_save(up, lat_col, lon_col, value_col=value_col,
                                      rows_per_part=int(rows_per_part), use_parquet=use_parquet)
        st.info(f"一時ディレクトリに {n_parts} パートを保存しました: {tmpdir}")

    # ストリーミングで抽出（ここがアップロード対策の中核）
    try:
        work = _stream_sample_points(up, value_col=value_col, lon_col=lon_col, lat_col=lat_col,
                                     keep_pct=sampling_pct)
    except Exception as e:
        progress.progress(100, text="エラー"); st.exception(e); return

    if work.empty:
        progress.progress(100, text="完了"); st.warning("日本域内に有効な点がありません。"); return
    if len(work) >= MAX_POINTS:
        st.info(f"データが非常に多いため、{MAX_POINTS:,} 点にサンプリングして表示しています。"
                "より正確な密度把握には『メッシュ表示』または『ヒートマップ』をご利用ください。")

    # breaks & colors（個点のみ）
    if draw_method == "個々の点をプロット" and value_col and "val" in work.columns:
        vmin = float(np.nanmin(work["val"]))
        vmax = float(np.nanmax(work["val"]))
        inner = st.session_state.get("ui_inner_bounds")
        colors = st.session_state.get("ui_colors")
        bins_local = len(colors) if colors else 5
        breaks = _build_breaks_auto(vmin, vmax, bins_local, inner)
        if not colors or len(colors) != bins_local:
            colors = _palette_to_bins(BASE_PALETTES[base_palette], bins_local)
    else:
        breaks = [0, 1]; colors = ["#E67814"]

    progress.progress(10, text="点データ準備中...")

    # -------- Folium 描画のみ -----------------------------------------------
    import folium
    from streamlit_folium import st_folium

    center = (float(work["lat"].mean()), float(work["lon"].mean()))
    fmap = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True, prefer_canvas=True)

    # ベースマップ
    if use_photo:
        folium.TileLayer(
            tiles="https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg",
            attr="地理院地図（航空写真）", name="GSI Photo", overlay=False, control=False, max_zoom=18
        ).add_to(fmap)
    else:
        folium.TileLayer(
            tiles="https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
            attr="地理院地図（淡色）", name="GSI Pale", overlay=False, control=False, max_zoom=18
        ).add_to(fmap)

    legend_html = ""

    if draw_method == "ヒートマップ":
        from folium.plugins import HeatMap
        points = work[["lat", "lon", "val"]].dropna().values.tolist() if "val" in work.columns \
                 else work[["lat", "lon"]].dropna().values.tolist()
        r, b = heat_level if heat_level else (15, 18)
        HeatMap(points, radius=r, blur=b, max_zoom=18).add_to(fmap)
        legend_html = _legend_for_heatmap()

    elif draw_method == "メッシュ表示":
        bins_mesh = 7
        mesh_palette = _make_gradient(mesh_min_color, mesh_max_color, bins_mesh)
        cell_km = mesh_cell_km if mesh_cell_km is not None else 2.0
        fc, thr = _mesh_geojson_by_cellsize(work, cell_km=cell_km, colors=mesh_palette, bins=bins_mesh)
        folium.GeoJson(
            data=_dumps(fc),
            name="mesh",
            style_function=lambda feat: {
                "fillColor": feat["properties"]["c"],
                "color": feat["properties"]["c"],
                "weight": 0.5,
                "fillOpacity": 0.55,
            },
            control=False, show=True,
        ).add_to(fmap)
        legend_html = _legend_for_mesh(thr, mesh_palette)

    else:  # 個々の点をプロット（Folium）
        feats = _features_with_color(work, breaks, colors, progress)
        from collections import defaultdict
        BATCH = 50_000
        total = len(feats); done = 0
        for s in range(0, total, BATCH):
            batch = feats[s:s+BATCH]
            groups = defaultdict(list)
            for f in batch:
                groups[f["properties"]["c"]].append(f)
            for col, items in groups.items():
                fc = {"type": "FeatureCollection", "features": items}
                folium.GeoJson(
                    data=_dumps(fc),
                    name=f"p_{s}_{col}",
                    marker=folium.CircleMarker(
                        radius=point_radius,
                        color=col, fill=True, fill_color=col, fill_opacity=0.7, weight=0
                    ),
                    control=False, show=True,
                ).add_to(fmap)
            done = min(total, s + BATCH)
            progress.progress(95 + int(4 * done / total), text=f"地図を描画中... {done:,}/{total:,}")
        legend_html = _legend_for_points(breaks, colors)

    # 表示
    map_slot.empty()
    st_folium(fmap, width=None, height=640, returned_objects=[], key=f"map-{np.random.randint(1e9)}")

    # DL（Foliumのみ）
    html = fmap.get_root().render()
    download_slot.empty()
    download_slot.download_button(
        "地図をHTML形式でダウンロード",
        data=html.encode("utf-8"),
        file_name="map.html",
        mime="text/html",
    )

    if legend_html:
        legend_slot.markdown("**凡例：**")
        legend_slot.markdown(legend_html, unsafe_allow_html=True)

    progress.progress(100, text="完了")

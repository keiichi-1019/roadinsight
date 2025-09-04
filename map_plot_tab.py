import json
import math
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

# --- optional orjson ---------------------------------------------------------
try:
    import orjson
    def _dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def _dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

# --- constants ---------------------------------------------------------------
EXAMPLE_DIR = "assets/map_examples"
PLOT_IMG = f"{EXAMPLE_DIR}/Plot_display_example.png"
MESH_IMG = f"{EXAMPLE_DIR}/Mesh_display_example.png"
HEAT_IMG = f"{EXAMPLE_DIR}/Heat_map_display_example.png"

# ---------- small utils ------------------------------------------------------
def _read_first_csv(file):
    name = file.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(file) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    with zf.open(n) as f:
                        return pd.read_csv(f), f"{file.name} / {n}"
        return None, None
    return pd.read_csv(file), file.name

def _idx_to_excel_col(idx: int) -> str:
    s, n = "", idx
    while True:
        n, r = divmod(n, 26)
        s = chr(65 + r) + s
        if n == 0:
            break
        n -= 1
    return s

def _excel_label_map(cols):
    """{'A': colname0, 'B': colname1, ...} を返す"""
    return {_idx_to_excel_col(i): c for i, c in enumerate(cols)}

def _score_in_range(series, low, high):
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

# ---------- preprocessing -----------------------------------------------------
def _make_work_df(
    df: pd.DataFrame,
    value_col: str | None,
    lon_override: str | None = None,
    lat_override: str | None = None
):
    # オーバーライド優先（未指定時は自動検出）
    if lon_override and lat_override:
        lon_col, lat_col = lon_override, lat_override
    else:
        lon_col, lat_col = _autodetect_lon_lat(df)
        if (lon_override and lon_override in df.columns):
            lon_col = lon_override
        if (lat_override and lat_override in df.columns):
            lat_col = lat_override

    if lon_col is None or lat_col is None:
        raise ValueError("緯度・経度列を特定できません。手動で指定してください。")

    cols = [lat_col, lon_col]
    if value_col and value_col not in cols:
        cols.append(value_col)

    w = df[cols].copy().rename(columns={lat_col: "lat", lon_col: "lon"})
    w["lat"] = pd.to_numeric(w["lat"], errors="coerce")
    w["lon"] = pd.to_numeric(w["lon"], errors="coerce")
    if value_col:
        w["val"] = pd.to_numeric(w[value_col], errors="coerce")
    w = w.dropna(subset=["lat", "lon"])
    w = w[w["lat"].between(20, 46) & w["lon"].between(122, 154)]
    if w.empty:
        raise ValueError("日本域（lat:20–46, lon:122–154）内に有効な点がありません。列の指定を確認してください。")
    # 軽量化
    w["lat"] = w["lat"].astype("float32").round(5)
    w["lon"] = w["lon"].astype("float32").round(5)
    if "val" in w.columns:
        w["val"] = w["val"].astype("float32")
    return w

# ---------- palettes & breaks -------------------------------------------------
BASE_PALETTES = {
    "Red→Blue": ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"],  # 小→大で赤→青
    "Blue→Red": ["#4575b4", "#91bfdb", "#fee090", "#fc8d59", "#d73027"],
    "Viridis":  ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151"],
    "Plasma":   ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636"],
    "Cividis":  ["#00204d", "#2b4066", "#566979", "#8f9c7f", "#d4e06b"],
    "Greys":    ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"],  # 参考
}

def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def _rgb_to_hex(rgb): return "#%02x%02x%02x" % rgb
def _lerp(a, b, t): return tuple(int(round(a[i] + (b[i]-a[i]) * t)) for i in range(3))

def _palette_to_bins(base_colors, bins: int):
    """基礎パレットを等間隔補間して bins 色を生成"""
    if bins <= 1:
        return [base_colors[0]]
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

def _build_breaks_auto(vmin: float, vmax: float, bins: int, inner_bounds: list[float] | None):
    """vmin/vmax は自動推定。inner_bounds は bins-1 個（任意）。"""
    if inner_bounds and len(inner_bounds) == bins - 1:
        bounds = sorted(float(x) for x in inner_bounds)
    else:
        bounds = list(np.linspace(vmin, vmax, bins + 1)[1:-1])
    return [float(vmin)] + bounds + [float(vmax)]

# ---------- features for point plotting --------------------------------------
def _features_with_color(work: pd.DataFrame, breaks, colors, progress):
    coords = np.column_stack((work["lon"].to_numpy(), work["lat"].to_numpy()))
    has_val = "val" in work.columns
    vals = work["val"].to_numpy() if has_val else None

    import bisect
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

# ---------- sampling ----------------------------------------------------------
def _apply_sampling(df: pd.DataFrame, pct: float, seed: int | None = 0) -> pd.DataFrame:
    """0.01–100% で行サンプリングを実施（5段階UIを想定）"""
    pct = max(0.01, min(100.0, float(pct)))
    if pct >= 99.999 or len(df) == 0:
        return df
    n = int(round(len(df) * pct / 100.0))
    if n <= 0:
        return df.iloc[0:0]
    return df.sample(n=n, random_state=seed)

# ---------- mesh helpers ------------------------------------------------------
def _estimate_deg_for_km(cell_km: float, center_lat: float):
    """km を中心緯度の度数に換算（緯度は約 111km/deg、経度は 111*cosφ km/deg）"""
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
    """
    点群を lon/lat の 2D ヒストグラムに集計し、セルサイズ[km]でビンを決めて
    各ビンをポリゴンで表現する GeoJSON を返す。colors は濃淡の配列。
    """
    min_lon, max_lon = float(work["lon"].min()), float(work["lon"].max())
    min_lat, max_lat = float(work["lat"].min()), float(work["lat"].max())
    lat0 = float(work["lat"].mean())
    dlon, dlat = _estimate_deg_for_km(cell_km, lat0)

    # 余白を少し
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
    counts = H.T  # shape (ny, nx)

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
            y0, y1 = yedges[iy], yedges[iy+1]
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            feats.append({
                "type": "Feature",
                "properties": {"c": col, "count": float(v)},
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            })
    return {"type": "FeatureCollection", "features": feats}, thr

# ---------- legend helpers ----------------------------------------------------
def _legend_block(items):
    # items : list of (label, color_hex)
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
        if i == 0:
            label = f"[{lo:.2g}, {hi:.2g}]"
        else:
            label = f"({lo:.2g}, {hi:.2g}]"
        items.append((label, colors[i]))
    return _legend_block(items)

def _legend_for_mesh(thr, colors):
    # thr: len==bins-1
    bounds = [0.0] + [float(x) for x in thr]
    items = []
    for i, col in enumerate(colors):
        lo = bounds[i]; hi = bounds[i+1] if i < len(thr) else "max"
        if hi == "max":
            label = f"({lo:.2g}, max]"
        elif i == 0:
            label = f"[>0, {hi:.2g}]"
        else:
            label = f"({lo:.2g}, {hi:.2g}]"
        items.append((label, col))
    return _legend_block(items)

def _legend_for_heatmap():
    # HeatMap の凡例（簡易説明）
    html = _legend_block([
        ("低密度", "#ffffb2"),
        ("中密度", "#fd8d3c"),
        ("高密度", "#bd0026")
    ])
    return html

# ---------- main UI ----------------------------------------------------------
def show_map_plot_tab():
    st.header("地図にプロット")

    # タイトル右に「表示例を見る」ポップアップ
    c1, c2 = st.columns([1, 0.25])
    with c1:
        st.write("緯度経度を持つデータを地図にプロットできます")
    with c2:
        with st.popover("表示例を見る（クリック）"):
            st.caption("プロット・メッシュ・ヒートマップの表示例")
            tabs = st.tabs(["個々の点をプロット", "メッシュ表示", "ヒートマップ"])
            with tabs[0]:
                st.image(PLOT_IMG, use_container_width=True)
            with tabs[1]:
                st.image(MESH_IMG, use_container_width=True)
            with tabs[2]:
                st.image(HEAT_IMG, use_container_width=True)

    st.write("ZIP/CSVファイルをアップロードしてください（複数選択できません）")
    st.write("※各ファイル最大30GBまで対応（ただし容量の大きなファイルは処理できない可能性があります）")

    up = st.file_uploader(
        "",
        type=["zip", "csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if not up:
        st.info("ファイルをアップロードしてください。")
        return

    # --- 入力読込 -----------------------------------------------------------
    try:
        df, src = _read_first_csv(up)
    except Exception as e:
        st.exception(e); return
    if df is None or df.empty:
        st.warning("有効なデータが見つかりません"); return

    # プレビュー（10 行）＋Excel列ラベル＋自動判定列を着色
    st.caption(f"プレビュー: {src}")
    label_map = _excel_label_map(df.columns)
    preview = df.head(10).copy()
    preview.columns = list(label_map.keys())

    auto_lon, auto_lat = _autodetect_lon_lat(df)
    lat_label_auto = next((k for k, v in label_map.items() if v == auto_lat), None)
    lon_label_auto = next((k for k, v in label_map.items() if v == auto_lon), None)

    def _style_preview(sdf: pd.DataFrame):
        styles = pd.DataFrame("", index=sdf.index, columns=sdf.columns)
        if lat_label_auto in sdf.columns:
            styles[lat_label_auto] = "background-color: #FFF7CC"  # 緯度＝淡黄色
        if lon_label_auto in sdf.columns:
            styles[lon_label_auto] = "background-color: #CCE5FF"  # 経度＝淡青色
        return styles
    st.dataframe(preview.style.apply(_style_preview, axis=None))

    # ---- 緯度/経度 手動指定（列ラベルから選択） ----
    with st.expander("緯度・経度の列が自動判定されない場合はこちらを設定",
                     expanded=(auto_lon is None or auto_lat is None)):
        labels = list(label_map.keys())
        c1, c2 = st.columns(2)
        lat_label_sel = c1.selectbox(
            "緯度の列を指定してください",
            options=["自動検出"] + labels,
            index=(labels.index(lat_label_auto) + 1) if lat_label_auto in labels else 0
        )
        lon_label_sel = c2.selectbox(
            "経度の列を指定してください",
            options=["自動検出"] + labels,
            index=(labels.index(lon_label_auto) + 1) if lon_label_auto in labels else 0
        )
        lat_override = None if lat_label_sel == "自動検出" else label_map[lat_label_sel]
        lon_override = None if lon_label_sel == "自動検出" else label_map[lon_label_sel]

    # ---- 描画オプション -----------------------------------------------------
    st.subheader("描画オプション")

    # 背景地図：淡色地図（既定）／航空写真に切替
    use_photo = st.checkbox("ベースマップを航空写真にする", value=False)

    # サンプリング：5段階固定
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

    # 値で色分け（個々の点のみ表示）
    use_color = False
    bins = 5
    base_palette = "Red→Blue"
    point_radius = 2.2  # 追加: 点サイズ
    if draw_method == "個々の点をプロット":
        st.markdown("**値で色分け**")
        use_color = st.checkbox("値で色分けする", value=False)
        point_radius = st.slider(
            "点のサイズ（半径・px）", min_value=1.0, max_value=8.0, value=2.2, step=0.2,
            help="プロットする点（CircleMarker）の半径（ピクセル）"
        )
        if use_color:
            labels = list(label_map.keys())
            value_label = st.selectbox("色分けに使う列ラベルを選択してください", options=labels, index=0)
            value_col = label_map[value_label]

            s = pd.to_numeric(df[value_col], errors="coerce")
            vmin_auto = float(s.min(skipna=True)); vmax_auto = float(s.max(skipna=True))

            c1, c2 = st.columns([1, 2])
            bins = int(c1.number_input("レンジ数", min_value=3, max_value=9, value=5, step=1))
            base_palette = c2.selectbox("使用するカラーセット", list(BASE_PALETTES.keys()), index=0)

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
        value_col = None

    # メッシュ表示：セルサイズ 10段階 & 2色指定（最小～最大）
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
        sel = m1.select_slider("メッシュサイズ", options=size_labels, value=size_labels[3],
                               help="最大10km～最小100mまでの10段階。")
        mesh_cell_km = size_values[size_labels.index(sel)]
        mesh_min_color = m2.color_picker("メッシュ最小レンジ色", value=mesh_min_color)
        mesh_max_color = st.color_picker("メッシュ最大レンジ色", value=mesh_max_color)

    # ヒートマップ：クラスタリング 5段階（わかりやすい表現）
    heat_level = None
    if draw_method == "ヒートマップ":
        level_label = st.select_slider(
            "密度の見せ方",
            options=["細かい", "やや細かい", "普通", "やや粗い", "粗い"],
            value="普通",
            help="左ほど局所的（小さな半径）、右ほど広域的（大きな半径）にぼかして表示。"
        )
        lvl_map = {
            "細かい": (6, 8),
            "やや細かい": (10, 12),
            "普通": (15, 18),
            "やや粗い": (20, 24),
            "粗い": (30, 36),
        }
        heat_level = lvl_map[level_label]

    # ---- session init -------------------------------------------------------
    if "work_df" not in st.session_state:
        st.session_state["work_df"] = None
    if "features" not in st.session_state:
        st.session_state["features"] = None
    if "last_map_html" not in st.session_state:
        st.session_state["last_map_html"] = None
    if "render_id" not in st.session_state:
        st.session_state["render_id"] = 0

    # ★ 表示順を固定：ボタン → 進捗 → 地図 → DL → 凡例
    start = st.button("地図を描画する")
    progress_slot = st.empty()
    map_slot = st.empty()
    download_slot = st.empty()
    legend_slot = st.empty()

    if start:
        progress = progress_slot.progress(0, text="前処理中...")

        # 値色分け列（個点時のみ）
        if draw_method == "個々の点をプロット" and use_color and st.session_state.get("color_value_label"):
            value_col = label_map[st.session_state["color_value_label"]]
        else:
            value_col = None

        # 前処理
        try:
            work = _make_work_df(df, value_col, lon_override=lon_override, lat_override=lat_override)
        except Exception as e:
            progress.progress(100, text="エラー"); st.exception(e); return

        # サンプリング
        work = _apply_sampling(work, sampling_pct, seed=0)
        st.session_state["work_df"] = work

        # breaks & colors（個点のみ）
        if draw_method == "個々の点をプロット" and value_col:
            vmin = float(st.session_state.get("ui_vmin_auto", float(np.nanmin(work["val"]))))

            vmax = float(st.session_state.get("ui_vmax_auto", float(np.nanmax(work["val"]))))

            inner = st.session_state.get("ui_inner_bounds")
            colors = st.session_state.get("ui_colors")
            bins_local = len(colors) if colors else 5
            breaks = _build_breaks_auto(vmin, vmax, bins_local, inner)
            if not colors or len(colors) != bins_local:
                colors = _palette_to_bins(BASE_PALETTES[base_palette], bins_local)
        else:
            breaks = [0, 1]; colors = ["#E67814"]

        progress.progress(10, text="点データ準備中...")

        # -------- Folium 描画 ------------------------------------------------
        import folium
        from streamlit_folium import st_folium

        center = (float(work["lat"].mean()), float(work["lon"].mean()))
        fmap = folium.Map(location=center, zoom_start=11, tiles=None,
                          control_scale=True, prefer_canvas=True)

        # ベースマップ：淡色地図 / 航空写真
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

        legend_html = ""  # 後で設定

        if draw_method == "ヒートマップ":
            from folium.plugins import HeatMap
            if "val" in work.columns:
                points = work[["lat", "lon", "val"]].dropna().values.tolist()
            else:
                points = work[["lat", "lon"]].dropna().values.tolist()
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

        else:  # 個々の点をプロット
            feats = _features_with_color(work, breaks, colors, progress)
            st.session_state["features"] = feats
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
            # 凡例
            legend_html = _legend_for_points(breaks, colors)

        # 表示：プログレスバー → 地図 → DLボタン → 凡例
        map_slot.empty()
        st.session_state["render_id"] += 1
        render_key = f"map-{st.session_state['render_id']}"
        with map_slot:
            st_folium(fmap, width=None, height=640, returned_objects=[], key=render_key)

        html = fmap.get_root().render()
        st.session_state["last_map_html"] = html

        download_slot.empty()
        with download_slot:
            st.download_button(
                "地図をHTML形式でダウンロード",
                data=html.encode("utf-8"),
                file_name="map.html",
                mime="text/html",
                key=f"dl-{st.session_state['render_id']}",
            )

        if legend_html:
            legend_slot.markdown("**凡例：**", help="色の範囲を示します。")
            legend_slot.markdown(legend_html, unsafe_allow_html=True)

        progress.progress(100, text="完了")

    # 再訪時は何も描画しない（直前の結果はボタン押下でのみ更新）
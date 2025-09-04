import streamlit as st
from common_tab import show_common_tab
from map_plot_tab import show_map_plot_tab
from mosaic_tab import show_mosaic_tab
from double_link_tab import show_double_link_tab
from route_map_tab import show_route_map_tab  # ← ファイル名・関数名ともに正確に

def main():
    st.set_page_config(page_title="ETC2.0プローブデータ加工ツール", layout="wide")
    st.title("ETC2.0プローブデータ加工ツール")

    tabs = ["データ結合・抽出", "地図にプロット", "モザイク図", "ダブルリンク図", "経路図"]
    selected = st.tabs(tabs)

    with selected[0]:
        show_common_tab()
    with selected[1]:
        show_map_plot_tab()
    with selected[2]:
        show_mosaic_tab()
    with selected[3]:
        show_double_link_tab()
    with selected[4]:
        show_route_map_tab()  # ← ここも「_map_」を忘れずに！

if __name__ == "__main__":
    main()

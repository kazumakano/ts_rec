import cv2
import numpy as np
import requests as req
import streamlit as st
import script.utility as util

URI = "http://192.168.175.120/action/snap?cam=0&user=admin&pwd=12345"

def render() -> None:
    st.title("Position Checker")

    for c, i in zip(st.columns(6), util.extract_ts_fig(cv2.imdecode(np.frombuffer(req.get(URI).content, dtype=np.uint8), cv2.IMREAD_COLOR))):
        with c:
            st.image(i, use_column_width=True, channels="BGR")

if __name__ == "__main__":
    render()

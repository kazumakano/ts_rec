import os
import os.path as path
from datetime import date
from glob import glob, iglob
import cv2
import numpy as np
import streamlit as st
import script.utility as util

SRC_ROOT_DIR = "/mnt/qnap105/"
TGT_ROOT_DIR = "/mnt/bigdata/01_projects/ts_fig_dataset/"

def _back_states() -> None:
    st.session_state["digit"] -= 1
    if st.session_state["digit"] == -1:
        st.session_state["vid_idx"] -= 1
        st.session_state["digit"] = 5

def _next_states() -> None:
    st.session_state["digit"] += 1
    if st.session_state["digit"] == 6:
        st.session_state["vid_idx"] += 1
        st.session_state["digit"] = 0

def _reset_states() -> None:
    st.session_state["vid_idx"] = 0
    st.session_state["digit"] = 0

def _check_exist(cam_name: str, vid_date: date) -> None:
    while True:
        if len(glob(path.join(TGT_ROOT_DIR, f"*_{vid_date.isoformat()}_{cam_name}_{st.session_state['vid_idx']:02d}_{st.session_state['digit']}_[0-9].tif"))) > 0:
            _next_states()
        else:
            break

def _save_ts_fig_img(cam_name: str, img: np.ndarray, label: int, usr_name: str, vid_date: date) -> None:
    file_name = f"{usr_name}_{vid_date.isoformat()}_{cam_name}_{st.session_state['vid_idx']:02d}_{st.session_state['digit']}_{label}.tif"
    if cv2.imwrite(path.join(TGT_ROOT_DIR, file_name), img):
        st.success(f"saved to {file_name}")
    else:
        st.error("failed to save image")

    _next_states()

def _label_btn(cam_name: str, img: np.ndarray, label: int, usr_name: str, vid_date: date) -> None:
    st.button(str(label), on_click=lambda: _save_ts_fig_img(cam_name, img, label, usr_name, vid_date))

def _undo(cam_name: str, vid_date: date) -> None:
    _back_states()

    files = glob(path.join(TGT_ROOT_DIR, f"*_{vid_date.isoformat()}_{cam_name}_{st.session_state['vid_idx']:02d}_{st.session_state['digit']}_[0-9].tif"))
    if len(files) > 0:
        os.remove(files[0])
        st.info(f"deleted {path.basename(files[0])}")

def _write_img_desc(file: str) -> None:
    match st.session_state["digit"]:
        case 0:
            digit_str = "1st"
        case 1:
            digit_str = "2nd"
        case 2:
            digit_str = "3rd"
        case other:
            digit_str = str(other + 1) + "th"

    st.write(f"showing {digit_str} figure on {file}")

def render() -> None:
    if len(st.session_state) == 0:
        _reset_states()

    st.title("Dataset Creator")

    usr_name = st.text_input("input your name")

    if usr_name != "":
        vid_date = st.date_input("choose date", on_change=_reset_states)
        cam_dir = st.selectbox("choose camera", [d for d in sorted(iglob(path.join(SRC_ROOT_DIR, str(vid_date.year), vid_date.isoformat(), "camera*")))], format_func=lambda dir: path.basename(dir)[6:], on_change=_reset_states)

        if cam_dir is not None:
            cam_name = path.basename(cam_dir)[6:]
            _check_exist(cam_name, vid_date)

            files = sorted(iglob(path.join(cam_dir, f"video_??-??-??_{st.session_state['vid_idx']:02d}.mkv")))
            if len(files) == 0:
                st.write("labeling completed")
            else:
                _write_img_desc(files[0])

                img = util.extract_ts_fig(util.read_head_n_frms(files[0], 1).squeeze())[st.session_state["digit"]]
                st.image(img, width=256, channels="BGR")

                for i, c in enumerate(st.columns(10)):
                    with c:
                        _label_btn(cam_name, img, i, usr_name, vid_date)

            cols = st.columns(8)
            with cols[0]:
                st.button("undo", on_click=lambda: _undo(cam_name, vid_date), disabled=st.session_state["vid_idx"] == 0 and st.session_state["digit"] == 0)
            with cols[1]:
                st.button("skip", on_click=_next_states)

if __name__ == "__main__":
    render()

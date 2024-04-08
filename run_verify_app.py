import base64
import io
import os.path as path
from glob import glob, iglob
import cv2
import flask
import pandas as pd
from PIL import Image
import script.utility as util

RESULT_DIR = "/mnt/bigdata/00_students/kazuma_nis/ts_rec/result/foo/"
VER = 0
VID_DIR = "/mnt/gazania/trusco-mp4/2024/2024-01-01/"

app = flask.Flask(__name__)

def init() -> None:
    global file_idx, slice_idx

    file_idx, slice_idx = 0, 0

@app.route("/")
def get() -> tuple[str, int]:
    global file_idx, slice_idx

    result_files = sorted(iglob(path.join(RESULT_DIR, f"*/??-??-??_*/version_{VER}/predict_results.csv")))

    while file_idx < len(result_files):
        result = pd.read_csv(result_files[file_idx], usecols=("cam", "vid_idx", "frm_idx", "recog", "is_inconsis"))
        slices_around_inconsis = util.slice_around_inconsis(result)

        if slice_idx == len(slices_around_inconsis):
            file_idx, slice_idx = file_idx + 1, 0
            continue

        cap = cv2.VideoCapture(filename=glob(path.join(VID_DIR, f"camera{result.loc[0, 'cam']}/video_{result_files[file_idx].split('/')[-3]}.mp4"))[0])    # support for video index duplication
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) != len(result):
            raise Exception("number of video frames and length of prediction result do not match")

        data, next_frm_idx = [], 0
        for _, r in slices_around_inconsis[slice_idx].iterrows():
            while next_frm_idx < r["frm_idx"]:
                cap.read()
                next_frm_idx += 1

            data.append({"frm_idx": r["frm_idx"], "imgs": [], "recogs": []})
            for i, img in enumerate(util.extract_ts_fig(cap.read()[1])):
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
                data[-1]["imgs"].append(base64.b64encode(buf.getvalue()).decode())
                data[-1]["recogs"].append(r["recog"][i + i // 2])
            next_frm_idx += 1

        cap.release()

        return flask.render_template(
            "index.html",
            data=data,
            result_file=glob(path.join(RESULT_DIR, f"*/??-??-??_*/version_{VER}/predict_results.csv"))[file_idx]
        ), 200

    return "verification completed", 200

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="specify server host", metavar="HOST")
    parser.add_argument("--port", default=5000, type=int, help="specify server port", metavar="PORT")
    args = parser.parse_args()

    init()
    app.run(host=args.host, port=args.port)

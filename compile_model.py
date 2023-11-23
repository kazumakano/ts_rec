import os.path as path
from glob import glob
import torch
from script.model import CNN3


def compile_model(result_dir: str, ver: int = 0) -> None:
    CNN3.load_from_checkpoint(
        glob(path.join(result_dir, f"version_{ver}/", "checkpoints/", "epoch=*-step=*.ckpt"))[0],
        loss_weight=torch.empty(10, dtype=torch.float32)
    ).to_torchscript(
        file_path=path.join(result_dir, f"version_{ver}/", "model.pt"),
        method="trace",
        example_inputs=torch.empty(1, 3, 22, 17, dtype=torch.float32)
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", required=True, help="specify train result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-v", "--ver", default=0, type=int, help="specify version", metavar="VER")
    args = parser.parse_args()

    compile_model(args.result_dir, args.ver)

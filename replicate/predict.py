"""Cog predictor for selective face anonymization (Replicate: divide-by-0/deface-selective).

Wraps deface/main.py: takes a video plus 1+ reference images of the person to KEEP
unblurred, runs the selective-anonymization pipeline, returns the anonymized video.

Place this at the repo root for the build (cog.yaml predict: "predict.py:Predictor"),
with the deface/ package + weights (yolo11x.pt, models/osnet_ms_d_c.pth.tar,
deface/centerface.onnx) in the build context. Apply the main.py import fixes from
DEPLOYMENT_REPLICATE.md first.
"""
import os
import shutil
import subprocess
import tempfile
from typing import List

from cog import BasePredictor, Input, Path

SRC = "/src"


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        video: Path = Input(description="Input video to anonymize (URL or file)"),
        keep_person_images: List[Path] = Input(
            description="One or more reference images of the person to KEEP unblurred "
            "(more angles = better tracking). Everyone else is anonymized."
        ),
        replacewith: str = Input(
            description="Anonymization filter", default="blur",
            choices=["blur", "solid", "mosaic", "none"],
        ),
        thresh: float = Input(description="Face detection threshold", default=0.4),
        reid_threshold: float = Input(
            description="Target-person re-id similarity threshold", default=0.7
        ),
        keep_audio: bool = Input(description="Keep original audio", default=True),
    ) -> Path:
        workdir = tempfile.mkdtemp(dir="/tmp")
        job = os.path.join(workdir, "job")
        tp = os.path.join(job, "target_person")
        os.makedirs(tp, exist_ok=True)
        shutil.copy(str(video), os.path.join(job, "video.mp4"))
        for idx, img in enumerate(keep_person_images):
            shutil.copy(str(img), os.path.join(tp, "ref_%d.jpg" % idx))

        env = dict(os.environ)
        env["PYTHONPATH"] = SRC + ":" + os.path.join(SRC, "deface")
        cmd = [
            "python", "deface/main.py", workdir,
            "--video-filename", "video.mp4",
            "--target-person-dirname", "target_person",
            "--replacewith", replacewith,
            "--thresh", str(thresh),
            "--reid-threshold", str(reid_threshold),
        ]
        if keep_audio:
            cmd.append("--keep-audio")
        subprocess.run(cmd, cwd=SRC, env=env, check=True)

        out = os.path.join(job, "anonymized_video.mp4")
        if not os.path.exists(out):
            raise RuntimeError("Anonymization produced no output")
        final = os.path.join(workdir, "anonymized.mp4")
        shutil.move(out, final)
        return Path(final)

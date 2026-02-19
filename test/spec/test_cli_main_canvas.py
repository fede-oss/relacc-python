import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path, rows):
    path.write_text("\n".join(rows), encoding="utf-8")


def _sample_rows(offset=0):
    return [
        "stroke_id x y time is_writing",
        f"0 {10+offset} 20 0 1",
        f"0 {12+offset} 22 10 1",
        f"1 {14+offset} 24 20 1",
    ]


def test_main_canvas_stdout_img_mode(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    _write_csv(f1, _sample_rows(0))

    res = subprocess.run(
        [sys.executable, str(ROOT / "main-canvas.py"), "-r", "3", str(f1)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<img src=\"data:image/png;base64," in res.stdout


def test_main_canvas_all_file_formats(tmp_path):
    f1 = tmp_path / "s1-arrow-t1.csv"
    f2 = tmp_path / "s1-arrow-t2.csv"
    _write_csv(f1, _sample_rows(0))
    _write_csv(f2, _sample_rows(1))

    for ext in ["png", "jpg", "pdf", "svg"]:
        out = tmp_path / f"out.{ext}"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "main-canvas.py"),
                "-r",
                "3",
                "-m",
                "centroid",
                "-o",
                str(out),
                str(f1),
                str(f2),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert out.exists()
        assert out.stat().st_size > 0

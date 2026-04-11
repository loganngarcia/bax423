#!/usr/bin/env python3
"""
Enhance speech in a video or audio file using MLX on Apple Silicon (mlx-audio).

Dependencies: pip install mlx-audio
System: ffmpeg on PATH (brew install ffmpeg)

Backends:
  deepfilter — mlx-community/DeepFilterNet-mlx (default, v3, good noise suppression)
  mossformer — starkdmi/MossFormer2_SE_48K_MLX (chunked for long clips)

Example:
  ./homework\\ 1/.venv/bin/python enhance_video_audio_mlx.py \\
    -i ~/Screenshots/foo_compressed.mp4 -o ~/Screenshots/foo_enhanced.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_ffmpeg(args: list[str]) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found. Install with: brew install ffmpeg")
    subprocess.run([exe, "-hide_banner", "-loglevel", "warning", "-y", *args], check=True)


VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpeg", ".mpg"}


def extract_audio_48k_mono_wav(media: Path, out_wav: Path) -> None:
    """48 kHz mono WAV for DeepFilterNet; MossFormer also resamples to 48 kHz internally."""
    args = ["-i", str(media)]
    if media.suffix.lower() in VIDEO_EXT:
        args.append("-vn")
    args += ["-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", str(out_wav)]
    run_ffmpeg(args)


def mux_video_with_audio(
    video_in: Path, wav_in: Path, video_out: Path, audio_codec: str
) -> None:
    if audio_codec == "copy":
        a_args = ["-c:a", "copy"]
    else:
        a_args = ["-c:a", "aac", "-b:a", "192k"]
    run_ffmpeg(
        [
            "-i",
            str(video_in),
            "-i",
            str(wav_in),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *a_args,
            "-shortest",
            str(video_out),
        ]
    )


def enhance_deepfilter(
    in_wav: Path,
    out_wav: Path,
    *,
    version: int,
    streaming: bool,
) -> None:
    from mlx_audio.sts import DeepFilterNetModel

    model = DeepFilterNetModel.from_pretrained("mlx-community/DeepFilterNet-mlx", version=version)
    if streaming:
        model.enhance_file_streaming(in_wav, out_wav)
    else:
        model.enhance_file(in_wav, out_wav)


def enhance_mossformer(in_wav: Path, out_wav: Path) -> None:
    from mlx_audio.sts import MossFormer2SEModel, save_audio

    model = MossFormer2SEModel.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")
    enhanced = model.enhance(str(in_wav))
    sr = getattr(model.config, "sample_rate", 48000)
    save_audio(enhanced, str(out_wav), sr)


def main() -> None:
    p = argparse.ArgumentParser(description="MLX speech enhancement for video or audio (Apple Silicon).")
    p.add_argument("-i", "--input", type=Path, required=True, help="Video (mp4/mov) or audio file")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output video or WAV path")
    p.add_argument(
        "--backend",
        choices=("deepfilter", "mossformer"),
        default="deepfilter",
        help="Enhancement model (default: deepfilter)",
    )
    p.add_argument("--df-version", type=int, default=3, choices=(1, 2, 3), help="DeepFilterNet version")
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="DeepFilterNet: load full file in memory (faster for short clips; may use more RAM)",
    )
    p.add_argument(
        "--audio-only",
        action="store_true",
        help="Write enhanced WAV only (no video remux). Output should end in .wav",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")

    suffix = inp.suffix.lower()
    is_audio_only_in = suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac"}
    if is_audio_only_in and out.suffix.lower() not in {".wav"} and not args.audio_only:
        sys.exit("Audio-only input: set output to a .wav file, or use --audio-only with a .wav path.")

    with tempfile.TemporaryDirectory(prefix="mlx_enhance_") as td:
        td_path = Path(td)
        raw_wav = td_path / "input_48k.wav"
        enhanced_wav = td_path / "enhanced.wav"

        print("Extracting 48 kHz mono WAV…", flush=True)
        extract_audio_48k_mono_wav(inp, raw_wav)

        print(f"Enhancing ({args.backend})…", flush=True)
        if args.backend == "deepfilter":
            use_stream = not args.no_streaming
            enhance_deepfilter(
                raw_wav,
                enhanced_wav,
                version=args.df_version,
                streaming=use_stream,
            )
        else:
            enhance_mossformer(raw_wav, enhanced_wav)

        if args.audio_only or out.suffix.lower() == ".wav":
            shutil.copy2(enhanced_wav, out)
            print(f"Wrote {out}", flush=True)
            return

        print("Muxing video with enhanced audio…", flush=True)
        mux_video_with_audio(inp, enhanced_wav, out, audio_codec="aac")
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()

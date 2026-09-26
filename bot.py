import json
import os
from pathlib import Path

import imageio_ffmpeg
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL


VIDEO_DIR = Path("videos")
TRANSCRIPT_DIR = Path("transcripts")

VIDEO_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)


def download_youtube_video(url: str) -> str:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    options = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(VIDEO_DIR / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_path,
        "merge_output_format": "mp4",
        "js_runtimes": {
            "node": {}
        },
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

        video_id = info["id"]
        requested_ext = info.get("ext", "mp4")

        possible_files = [
            VIDEO_DIR / f"{video_id}.mp4",
            VIDEO_DIR / f"{video_id}.{requested_ext}",
        ]

        for path in possible_files:
            if path.exists():
                return str(path)

    raise RuntimeError(f"Could not locate downloaded video for {url}")


def transcribe_video(video_path: str) -> str:
    print(f"Transcribing: {video_path}")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,
    )

    transcript = []

    for segment in segments:
        item = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        }

        transcript.append(item)

        print(
            f"[{item['start']:.2f} -> {item['end']:.2f}] "
            f"{item['text']}"
        )

    video_id = Path(video_path).stem
    output_path = TRANSCRIPT_DIR / f"{video_id}.json"

    result = {
        "video_id": video_id,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": transcript,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to {output_path}")

    return str(output_path)


def process_video(url: str):
    video_path = download_youtube_video(url)
    transcript_path = transcribe_video(video_path)

    return {
        "video_path": video_path,
        "transcript_path": transcript_path,
    }


if __name__ == "__main__":
    results = process_video(
        "https://www.youtube.com/watch?v=hN_ZUElLH44"
    )

    print(json.dumps(results, indent=2))
import json
import subprocess
from pathlib import Path


WLASL_JSON = Path(
    r"C:\Users\Admin\Downloads\DP Project\Datasets"
    r"\WLASL-master\start_kit\WLASL_v0.3.json"
)

RAW_DIR = Path("datasets/wlasl_raw")
CLIP_DIR = Path("datasets/wlasl_clips")

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SIGNS = {
    "hello",
    "please",
    "yes",
    "no",
    "sorry",
    "help",
    "good",
    "morning",
    "goodbye",
    "love",
    "you",
    "how",
}


MAX_SAMPLES_PER_SIGN = 20


with open(WLASL_JSON, "r", encoding="utf-8") as file:
    data = json.load(file)


def download_source(url, video_id):

    output = RAW_DIR / f"{video_id}.mp4"

    if output.exists():
        return output

    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "best[ext=mp4]/best",
        "-o",
        str(output),
        url,
    ]

    print(f"Downloading source: {video_id}")

    result = subprocess.run(command)

    if result.returncode != 0:
        return None

    return output


def crop_instance(source, output, instance):

    fps = instance.get("fps", 25)

    frame_start = instance.get("frame_start", 1)
    frame_end = instance.get("frame_end", -1)

    start_time = frame_start / fps

    if frame_end == -1:
        duration = None
    else:
        duration = (
            frame_end - frame_start
        ) / fps

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        str(start_time),
    ]

    if duration is not None:
        command.extend(
            [
                "-t",
                str(duration),
            ]
        )

    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            str(output),
        ]
    )

    result = subprocess.run(command)

    return result.returncode == 0


for entry in data:

    gloss = entry["gloss"].strip().lower()

    if gloss not in TARGET_SIGNS:
        continue

    sign_folder = (
        CLIP_DIR / gloss.upper()
    )

    sign_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("SIGN:", gloss.upper())
    print("=" * 60)

    saved_count = 0

    for instance in entry.get(
        "instances",
        []
    ):

        if saved_count >= MAX_SAMPLES_PER_SIGN:
            break

        url = instance.get("url")
        video_id = str(
            instance.get("video_id")
        )

        if not url:
            continue

        output_clip = (
            sign_folder /
            f"{gloss.upper()}_{video_id}.mp4"
        )

        if output_clip.exists():
            print(
                "Already exists:",
                output_clip.name,
            )

            saved_count += 1
            continue

        source = download_source(
            url,
            video_id,
        )

        if source is None:
            print(
                "FAILED download:",
                video_id,
            )
            continue

        success = crop_instance(
            source,
            output_clip,
            instance,
        )

        if success:
            saved_count += 1

            print(
                f"Saved {saved_count}:",
                output_clip,
            )

        else:
            print(
                "FAILED crop:",
                video_id,
            )

    print(
        f"{gloss.upper()} finished with "
        f"{saved_count} clips."
    )


print()
print("=" * 60)
print("SELECTED WLASL DOWNLOAD COMPLETE")
print("=" * 60)
print("Clips saved in:")
print(CLIP_DIR.resolve())
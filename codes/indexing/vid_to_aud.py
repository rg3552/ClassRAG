import argparse
import os
from moviepy.editor import VideoFileClip


def extract_audio_from_videos(video_dir, audio_dir):
    """
    Extracts audio from all video files in a directory and saves them as MP3 files.

    Parameters:
        video_dir (str): Directory containing video files.
        audio_dir (str): Directory where audio files will be saved.
    """
    # Ensure the audio directory exists
    os.makedirs(audio_dir, exist_ok=True)

    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        audio_path = os.path.join(audio_dir, video_filename.replace(".mp4", ".mp3"))

        # Check if the file is a video
        if not video_filename.endswith(".mp4"):
            print(f"Skipping non-video file: {video_filename}")
            continue

        print(f"Processing video: {video_path}")

        # Extract audio
        with VideoFileClip(video_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
            print(f"Audio saved to: {audio_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio from video files.")
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing video files."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory to save extracted audio files.",
    )

    args = parser.parse_args()

    extract_audio_from_videos(args.video_dir, args.audio_dir)


if __name__ == "__main__":
    main()

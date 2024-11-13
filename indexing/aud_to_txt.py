import argparse
import os
import whisper


def transcribe_audios(audio_dir, transcript_dir, model_name="medium"):
    """
    Transcribes audio files in a directory using a Whisper model and saves the transcriptions as text files.

    Parameters:
        audio_dir (str): Directory containing audio files.
        transcript_dir (str): Directory where transcriptions will be saved.
        model_name (str): Whisper model name to use for transcription.
    """
    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Ensure the transcript directory exists
    os.makedirs(transcript_dir, exist_ok=True)

    for audio_filename in os.listdir(audio_dir):
        if not audio_filename.endswith(".mp3"):
            print(f"Skipping non-audio file: {audio_filename}")
            continue

        audio_path = os.path.join(audio_dir, audio_filename)
        print(f"Transcribing audio: {audio_path}")

        # Transcribe audio
        result = model.transcribe(audio_path)

        # Save the transcription text to a file
        output_path = os.path.join(
            transcript_dir, audio_filename.replace(".mp3", ".txt")
        )
        with open(output_path, "w") as file:
            file.write(result["text"])

        print(f"Transcription saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files to text.")
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files."
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        required=True,
        help="Directory to save transcriptions.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="medium",
        help="Whisper model name to use (e.g., 'small', 'medium', 'large').",
    )

    args = parser.parse_args()

    transcribe_audios(args.audio_dir, args.transcript_dir, args.model_name)


if __name__ == "__main__":
    main()

import pandas as pd

import torch
import torchaudio
from pyannote.audio import Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_shortest_speaker(segments: pd.DataFrame):
    
    """
    Remove the speaker with the shortest total speaking time from the segments DataFrame.
    """

    shortest_speaker = segments.groupby("speaker")["duration"].sum().idxmin()
    segments = segments[segments["speaker"] != shortest_speaker]
    
    return segments

def diarize_audio(
    audio_path: str,
    diarization_model: str = "pyannote/speaker-diarization-3.1"):

    """
    Preprocess the audio file to obtain speaker diarization.
    Returns a pandas DataFrame with columns: ['start', 'end', 'speaker', 'duration']
    """

    info = torchaudio.info(audio_path)
    total_duration = info.num_frames / info.sample_rate

    # diarize the waveform
    pipeline = Pipeline.from_pretrained(diarization_model).to(device)
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker,
            'duration': turn.end - turn.start
        })

    segments = pd.DataFrame(segments)
    return segments, total_duration


def main():
    pass

if __name__ == "__main__":
    main()
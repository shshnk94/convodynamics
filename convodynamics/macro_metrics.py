import os
from argparse import ArgumentParser
from typing import Dict
import pandas as pd

from preprocess import diarize_audio, remove_shortest_speaker
from utils import adaptability, predictability
from feature import Feature

class SpeakingTime(Feature):

    def __init__(self):
        super().__init__(name="speaking_time")

    def extract(
        self, 
        conversation: pd.DataFrame,
        total_duration: float) -> Dict[str, float]:

        """
        Extract the speaking time for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their speaking times as values.
        """

        speaking_time = conversation.groupby("speaker")["duration"].sum() * 100 / total_duration

        return self._format_results(
            speaker_metrics={"speaking_time": speaking_time},
            speakers=conversation['speaker'].unique()
        )

class TurnLength(Feature):

    def __init__(self):
        super().__init__(name="turn_length")

    def extract(
        self, 
        conversation: pd.DataFrame) -> Dict[str, float]:

        """
        Extract the average turn length for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their average turn lengths as values.
        """

        median = conversation.groupby('speaker')['duration'].median()

        mean = conversation.groupby('speaker')['duration'].mean()
        cv = conversation.groupby('speaker')['duration'].std() / mean

        speakers = conversation['speaker'].unique()
        speaker_a, speaker_b = (
            conversation.loc[conversation['speaker'] == speakers[0], "duration"], 
            conversation.loc[conversation['speaker'] == speakers[1], "duration"]
        )        
        
        return self._format_results(
            speaker_metrics={
                "turn_length_median": median,
                "turn_length_mean": mean,
                "turn_length_cv": cv,
                "turn_length_predictability": pd.Series({
                    speakers[0]: predictability(speaker_a),
                    speakers[1]: predictability(speaker_b)
                })
            },
            conversation_metrics={"turn_length_adaptability": adaptability(speaker_a, speaker_b)},
            speakers=speakers
        )


class Pauses(Feature):

    def __init__(self):
        super().__init__(name="pauses")

    def extract(
        self, 
        conversation: pd.DataFrame, 
        total_duration: float):

        """
        Extract the average pause percentage for each participant in the conversation.
        """

        conversation['pause'] = conversation['start'].shift(-1) - conversation['end']
        
        avg_pause_pct = conversation.groupby('speaker')['pause'].mean().dropna() * 100 / total_duration
        speakers = conversation['speaker'].unique()
        
        return self._format_results(
            speaker_metrics={"avg_pause_pct": avg_pause_pct},
            speakers=speakers
    )

if __name__ == "__main__":

    parser = ArgumentParser(description="Extract speaking time feature from conversation data.")
    parser.add_argument("--datapath", type=str, help="Path to the folder with all conversations")
    args = parser.parse_args()

    ## Example with a random conversation ##
    random_conversation = "67836c1d-1334-41a0-a33a-4f788e8b6fb3"
    audio_path = os.path.join(
        args.datapath, 
        random_conversation, 
        f"processed/{random_conversation}.mp3"
    )
    segments, total_duration = diarize_audio(audio_path)

    ## Remove the shortest speaker ##
    segments = remove_shortest_speaker(segments)

    ## Setup all metrics ##
    speaking_time = SpeakingTime()(segments, total_duration=total_duration)
    turn_length = TurnLength()(segments)
    pauses = Pauses()(segments, total_duration=total_duration)

    print("Speaking Time: ", speaking_time)
    print("Turn Length: ", turn_length)
    print("Pauses: ", pauses)
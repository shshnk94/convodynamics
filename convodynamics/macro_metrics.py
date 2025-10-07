import os
from argparse import ArgumentParser

from base import Feature
from typing import Any, Dict

import pandas as pd

from preprocess import diarize_audio

class SpeakingTime(Feature):

    def __init__(self):
        super().__init__(name="speaking_time")

    def extract(
        self, 
        segments: pd.DataFrame,
        total_duration: float) -> Dict[str, float]:

        """
        Extract the speaking time for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their speaking times as values.
        """

        speaking_time = segments.groupby("speaker")["duration"].sum() * 100 / total_duration
        speaking_time = speaking_time.to_dict()
        return speaking_time

class TurnLength(Feature):

    def __init__(self):
        super().__init__(name="turn_length")

    def extract(
        self, 
        segments: pd.DataFrame) -> Dict[str, float]:

        """
        Extract the average turn length for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their average turn lengths as values.
        """

        median = segments.groupby('speaker')['duration'].median()
        mean = segments.groupby('speaker')['duration'].mean()
        cov = segments.groupby('speaker')['duration'].std() / mean
                
        turn_length = segments.groupby('speaker')['duration'].mean()
        turn_length = turn_length.to_dict()
        return turn_length

class Pauses(Feature):

    def __init__(self):
        super().__init__(name="pauses")

    def extract(
        self, 
        segments: pd.DataFrame,
        total_duration: float) -> Dict[str, float]:

        """
        Extract the average pause duration between turns for each participant in the conversation.
        Returns a dictionary with participant IDs as keys and their average pause durations as values.
        """

        segments = segments.sort_values(by='start')
        segments['pause'] = segments['start'].shift(-1) - segments['end']

        pauses = segments.groupby('speaker')['pause'].mean().dropna() * 100 / total_duration
        pauses = pauses.to_dict()
        return pauses

if __name__ == "__main__":

    parser = ArgumentParser(description="Extract speaking time feature from conversation data.")
    parser.add_argument("--datapath", type=str, help="Path to the folder with all conversations")
    args = parser.parse_args()

    ## Example with a random conversation ##
    random_conversation = "67836c1d-1334-41a0-a33a-4f788e8b6fb3"
    segments, total_duration = diarize_audio(random_conversation)

    # Drop the speaker with the shortest total duration
    speaker_durations = segments.groupby("speaker")["duration"].sum()
    shortest_speaker = speaker_durations.idxmin()
    segments = segments[segments["speaker"] != shortest_speaker]

    ## Setup all metrics ##
    speaking_time = SpeakingTime()(segments, total_duration)
    turn_length = TurnLength()(segments)
    pauses = Pauses()(segments, total_duration)

    print("Speaking Time: ", speaking_time)
    print("Turn Length: ", turn_length)
    print("Pauses: ", pauses)
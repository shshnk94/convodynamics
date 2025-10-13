import os
from argparse import ArgumentParser
from typing import Dict
import pandas as pd

from .utils import adaptability, predictability
from .feature import Feature

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
        conversation: pd.DataFrame, 
        **kwargs):

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

class SpeakerRate(Feature):

    def __init__(self):
        super().__init__(name="speaker_rate")

    def extract(
        self, 
        conversation: pd.DataFrame,
        text_field: str
    ):
    
        """
        Extract the speaking rate (words per minute) for each participant in the conversation.
        Since deciding on word boundaries is non-trivial, we use the provided transcript to count words.
        """

        def n_words(x):
            return len(x.split(' '))

        conversation["speechrate"] = conversation[text_field].apply(n_words) * 60 / conversation["delta"]

        median = conversation.groupby('speaker')['speechrate'].median()
        mean = conversation.groupby('speaker')['speechrate'].mean()
        cv = conversation.groupby('speaker')['speechrate'].std() / mean

        speakers = conversation['speaker'].unique()
        speaker_a, speaker_b = (
            conversation.loc[conversation['speaker'] == speakers[0], "speechrate"], 
            conversation.loc[conversation['speaker'] == speakers[1], "speechrate"]
        )

        return self._format_results(
            speaker_metrics={
                "speaker_rate_median": median,
                "speaker_rate_cv": cv,
                "speaker_rate_predictability": pd.Series({
                    speakers[0]: predictability(speaker_a),
                    speakers[1]: predictability(speaker_b)
                })
            },
            conversation_metrics={"speaker_rate_adaptability": adaptability(speaker_a, speaker_b)},
            speakers=speakers
        )

class Backchannels(Feature):

    def __init__(self):
        super().__init__(name="backchannels")

    def is_backchannel(self, current_segment, segments):

        check = any(
            # check if current segment is within another segment
            (segments["start"] <= current_segment["start"]) & \
            (segments["end"] >= current_segment["end"]) & \
            # check if current segment is <= 1 second
            (current_segment["end"] - current_segment["start"] <= 1.0)
        )

        return check

    def extract(
        self, 
        conversation: pd.DataFrame,
        **kwargs
    ):

        conversation['backchannel'] = conversation.apply(self.is_backchannel, args=(conversation,), axis=1)
        turns = conversation.groupby("speaker").size().astype(float)
        backchannels = conversation.groupby("speaker")["backchannel"].sum().astype(float)

        speakers = conversation["speaker"].unique()
        
        # defined as the proportion of the number of turns taken by the other speaker that are backchannels
        backchannels.loc[speakers[0]] = backchannels.loc[speakers[0]] * 100 / turns.loc[speakers[1]]
        backchannels.loc[speakers[1]] = backchannels.loc[speakers[1]] * 100 / turns.loc[speakers[0]]

        return self._format_results(
            speaker_metrics={"backchannels": backchannels},
            speakers=speakers
        )

class ResponseTime(Feature):

    def __init__(self):
        super().__init__(name="response_time")

    def extract(
        self, 
        conversation: pd.DataFrame,
        **kwargs
    ):

        conversation = conversation.sort_values(by="start").reset_index(drop=True)
        conversation['response_time'] = conversation['start'].shift(-1) - conversation['end']
        
        avg_response_time = conversation.groupby('speaker')['response_time'].mean().dropna()
        speakers = conversation['speaker'].unique()
        
        return self._format_results(
            speaker_metrics={"avg_response_time": avg_response_time},
            speakers=speakers
    )
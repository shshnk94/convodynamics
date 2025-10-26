from typing import Any, List, Dict
import pandas as pd
from convokit import Corpus
from convokit.transformer import Transformer

from .metrics import Feature

class ConversationDynamicsTransformer(Transformer):

    def __init__(self):

        pass

    def register_metrics(
        self,
        metrics: List[Feature]
    ) -> None:
        
        """
        Register a list of feature extraction metrics.
        """

        self.metrics = metrics

    def remove_shortest_speaker(
        self, 
        segments: pd.DataFrame
    ):
    
        """
        Remove the speaker with the shortest total speaking time from the segments DataFrame.
        """

        shortest_speaker = segments.groupby("speaker")["duration"].sum().idxmin()
        segments = segments[segments["speaker"] != shortest_speaker]
    
        return segments

    def transform(
        self,
        corpus: Corpus
    ):

        for conversation in corpus.iter_conversations():

            segments = conversation.retrieve_meta("diarization_segments")

            # if diarization segments are available, we will use them
            if segments is not None:

                # remove shortest speaker -- usually noise
                if segments['speaker'].nunique() > 2:
                    segments = self.remove_shortest_speaker(segments)

                metrics = {}
                for metric in self.metrics:

                    metric_name = metric.get_name
                    print("Extracting feature:", metric_name)
    
                    metrics[metric_name] = metric(
                        conversation=segments,
                        total_duration=segments['end'].max()
                    ) 

            # if no diarization segments, we will use the utterance transcripts
            else:

                transcripts = conversation.get_utterances_dataframe(exclude_meta=True)

                metrics = {}
                for metric in self.metrics:

                    metric_name = metric.get_name
                    print("Extracting feature:", metric_name)

                    metrics[metric_name] = metric(conversation=transcripts)

            print("Extracted features:", metrics)
            break
        
        return corpus
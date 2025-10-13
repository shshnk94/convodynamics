import argparse
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Union
import random
import pandas as pd

from .feature import Feature
from .preprocess import diarize_audio, load_transcript, remove_shortest_speaker
from .metrics import SpeakingTime, TurnLength, Pauses, SpeakerRate, Backchannels, ResponseTime

class FeatureExtractor:

    def __init__(
        self, 
        conversation_path: Union[str, Path] = None
    ):
        self.conversation_path = conversation_path if isinstance(conversation_path, Path) else Path(conversation_path)
        self.conversation_id = self.conversation_path.name

    def register_metrics(
        self,
        metrics: List[Feature]
    ) -> None:
        
        """
        Register a list of feature extraction metrics.
        """

        self.metrics = metrics

    def extract(self) -> Dict[str, Any]:

        """
        Extract all registered features from the conversation DataFrame.
        Returns a dictionary with feature names as keys and their extracted values.
        """
        
        features = {}
        for metric in self.metrics:

            metric_name = metric.get_name
            print("Extracting feature:", metric_name)

            if metric_name in ["speaker_rate"]:
                
                # load transcript
                if not hasattr(self, 'transcript'): 
                    self.transcript = load_transcript(self.conversation_path)

                # features[metric_name] = metric(conversation=self.transcript)
                pass
            
            else:

                # diarize the audio to get conversation segments
                if not hasattr(self, "segments"):
                    audio_path = self.conversation_path / "processed" / f"{self.conversation_id}.mp3"
                    self.segments, self.total_duration = diarize_audio(audio_path)

                # remove shortest speaker -- usually noise
                if self.segments['speaker'].nunique() > 2:
                    self.segments = remove_shortest_speaker(self.segments)

                features[metric_name] = metric(
                    conversation=self.segments,
                    total_duration=self.total_duration
                )

        return features

    def _combine_extracted(
        self, 
        extracted_features: Dict[str, Any]
    ) -> pd.DataFrame:

        """
        Format the extracted features into a pandas DataFrame for easier analysis.
        """

        combined_features = {}
        for name, features in extracted_features.items():
            for key, value in features.items():

                if isinstance(value, pd.Series):
                    for speaker, score in value.items():
                        combined_features[f"{feature_name}_{key}_{speaker}"] = score

                else:
                    combined_features[f"{feature_name}_{key}"] = value

        return pd.DataFrame([combined_features])
    
def main():
    
    parser = argparse.ArgumentParser(description="Extract conversational features from audio files.")
    parser.add_argument("--data_path", type=str, help="Path to the directory containing all conversations.")
    args = parser.parse_args()

    random.seed(42)
    data_path = Path(args.data_path)

    conversations = list(data_path.glob('*'))
    random_conversation = random.sample(conversations, 1)[0]

    print("Random conversation:", random_conversation)
    
    # Initialize feature extractor
    fe = FeatureExtractor(conversation_path=random_conversation)

    # Register desired metrics
    fe.register_metrics([
        SpeakingTime(),
        TurnLength(),
        Pauses(),
        SpeakerRate(),
        Backchannels(),
        ResponseTime()
    ])

    # Extract features
    features = fe.extract()

    print("Extracted features:", features)
    # # Combine and display results
    # features_df = extractor._combine_extracted(features)
    # print(features_df)

if __name__ == "__main__":

    main()
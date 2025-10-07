from abc import ABC, abstractmethod
from typing import Any, List, Dict

import pandas as pd

class Feature(ABC):

    def __init__(
        self, 
        name: str
    ):
        self._name = name

    @abstractmethod
    def extract(
        self, 
        conversation: Any, 
        **optional_kwargs
    ):
        """
        Extract the feature from the conversation data.
        This method should be implemented by all subclasses.
        """
        pass

    @property
    def get_name(self) -> str:
        return self._name

    def __call__(
        self, 
        conversation: Any, 
        **kwargs
    ):
        return self.extract(
            conversation=conversation, 
            **kwargs
        )

    def _format_results(
        self, 
        speaker_metrics: Dict[str, pd.Series] = None,
        conversation_metrics: Dict[str, float] = None,
        speakers: List[str] = None
    ):
        
        """
        Helper method to format results consistently across all feature extractors.
        
        Args:
            speaker_metrics: Dict with metric names as keys, Series with speaker data as values
            conversation_metrics: Dict with global metric names and their values
        """

        results = {}

        # Handle speaker-specific metrics
        if speaker_metrics:
            for metric, scores in speaker_metrics.items():
                #FIXME: check if all speakers are present in scores.index
                mapping = {s: f"{s.lower().strip()}_{metric}" for s in speakers}
                scores.index = scores.index.map(mapping)
                results.update(scores.to_dict())

        # Handle conversation-level metrics
        if conversation_metrics: results.update(conversation_metrics)
        
        return results
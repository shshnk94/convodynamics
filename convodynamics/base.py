from typing import Any, Dict, List

class Feature:

    def __init__(
        self, 
        name: str
    ):
        self._name = name

    def extract(
        self, 
        conversation: Any) -> Dict[str, Any]:
        """
        Extract feature(s) from a conversation object.
        Returns the feature(s) in a dictionary: could be scalar, vector, tensor, etc.
        """
        pass

    @property
    def get_name(self) -> str:
        return self._name

    def __call__(
        self, 
        conversation: Any) -> Dict[str, Any]:
        return self.extract(conversation)
from typing import List
from .Model import Model
from .SingleBlockRefiner import SingleBlockRefiner

class MultiBlockRefiner:    
    def __init__(self, model: Model, refiners: List[SingleBlockRefiner]):
        self.model = model
        self.refiners = refiners

    def __call__(self, query: str, text: str):
        for refiner in self.refiners:
            text = refiner(query, text)
        return text
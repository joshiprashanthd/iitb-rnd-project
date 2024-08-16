import os
from typing import List

from .Model import Model
from .utils import Question

from .Feedback import Feedback
from .RefinePoint import RefinePoint

class SingleBlockRefiner:
    def __init__(self, 
                 name: str, 
                 llm: Model, 
                 block_dir_path: str, 
                 refine_point_json_path: str = "/home/suraj/MedCoQ/newversion copy/data/misc/refine_point.json"):
        self.name = name
        self.llm = llm
        self.questions: List[Question] = []
        
        question_path = os.path.join(block_dir_path, "questions")
        
        self.refine_point_generator = RefinePoint(refine_point_json_path)
        
        json_paths = os.listdir(question_path)
        for path in json_paths:
            self.questions.append(Question(os.path.join(question_path, path)))

    def gen_feedbacks(self, query: str, text: str) -> List[str]: 
        feedbacks = []
        for q in self.questions:
            f = Feedback(q)(self.llm, query, text)
            feedbacks.append(f.strip())
        return feedbacks
    
    def gen_refined_text(self, text: str, feedbacks: List[str]):
        feedback = "\n".join(feedbacks)
        if len(feedback.strip()) > 0: 
            return self.refine_point_generator(self.llm, text, feedback)
        return text
    
    def __call__(self, query: str, text: str):
        feedbacks = self.gen_feedbacks(query, text)
        return self.gen_refined_text(text, feedbacks)    
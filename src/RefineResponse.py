from typing import List
from .utils import gen_points_text

class RefineResponse:
    def __init__(self) -> None:
        self.gen_prompt_from_examples()        
        
    def gen_prompt_from_examples(self):
        TEMPLATE = """Your task is to use the following points and make a combined response."""
        self.prompt = TEMPLATE
    
    def make_query(self, points: List[str]):
        query = f"""{self.prompt}
Points:
{gen_points_text(points)}
Combined Response: """
        return query

    def __call__(self, llm, points: List[str]) -> str:
        generation_query = self.make_query(points)
        response = llm.respond(generation_query)
        return response
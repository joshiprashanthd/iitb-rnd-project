import json, re
from .Model import Model

class RefinePoint:
    def __init__(self, file_path: str) -> None:
        self.point_prefix = "Point: "
        self.feedback_prefix = "Feedback: "
        self.refine_prefix = "Refined Point: "
        self.inter_example_sep = '\n'
        self.gen_prompt_from_examples(file_path)
        
    def gen_prompt_from_examples(self, file_path: str):
        TEMPLATE = """Your task is to read feedback carefully and follow the suggestions to do necessary corrections in the point.

Task Instructions:
- Refined point should contain human-like text.
- Refined point should retain the overall STRUCTURE and FLOW of the point as much as possible.
- Make only MINOR CHANGES to correct or improve the point according to the feedback.
- The number of words in refined point should NOT be much more than original point.
- Refined point should NOT contain unnecessary explanations or notes.
        
Learn from these examples how to use the feedback to refine the given point.

Examples:

{examples}"""

        EXAMPLE_TMP = """{point_prefix}{point}
{feedback_prefix}{feedback}
{refine_prefix}{refine_point}
"""

        examples = []
        with open(file_path, "r") as f:
            data = json.load(f)
            for e in data['examples'][:4]: # 4 shot prompt
                examples.append(EXAMPLE_TMP.format(feedback_prefix=self.feedback_prefix,
                                                   feedback=e['feedback'],
                                                   point_prefix=self.point_prefix,
                                                   point=e['point'],
                                                   refine_prefix=self.refine_prefix,
                                                   refine_point=e['refined']))
        
    
        self.prompt = TEMPLATE.format(examples=self.inter_example_sep.join(examples))
    
    def make_query(self, point: str, feedback: str):
        query = f"""{self.prompt}
Generate a refined point using the given point and feedback.
{self.point_prefix}{point}
{self.feedback_prefix}{feedback}"""
        return query

    def __call__(self, llm: Model, point: str, feedback: str) -> str:
        generation_query = self.make_query(point, feedback)
        response = llm.respond(generation_query, stop=['<\s>', '\n'])
        
        refined_point_regex = r"[\w|\s]*[(R|r)efined]*[\s]*[Pp]oint[:=-]\s*"
        explanation_regex = r"[^\w]*[Ee]xplanation[:=-].*"
        feedback_regex = r"[^\w]*[Ff]ee[d]?back[\s]*[:=-].*"

        refined = re.sub(refined_point_regex, "", response)
        refined = re.sub(feedback_regex, "", refined)
        refined = re.sub(explanation_regex, "", refined)
            
        # print(f"RefinePoint||{refined}")
        
        return refined
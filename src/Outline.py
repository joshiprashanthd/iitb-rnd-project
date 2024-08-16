from src.utils import gen_points_text
import json

class Outline:
    def __init__(self, file_path: str) -> None:
        self.query_prefix = "Query: "
        self.outline_prefix = "Outline: "
        self.gen_prompt_from_examples(file_path)
        
    def gen_prompt_from_examples(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
            
        TEMPLATE = """{instruction}

Learn from the examples to generate outline from the given query.

Examples:
{examples}
"""

        EXAMPLE_TEMP = """{query_prefix}{query}
{outline_prefix}
{points_text}"""

        
            
        
        examples = []
        for example in data['examples']:
            query = example['query']
            points_text = gen_points_text(example['outline'],numbered=True)
            examples.append(EXAMPLE_TEMP.format(query_prefix=self.query_prefix, 
                                                query=query, 
                                                outline_prefix=self.outline_prefix, 
                                                points_text=points_text))
        
        examples_text = "\n\n".join(examples)
        self.prompt = TEMPLATE.format(examples=examples_text, instruction=data['instruction'])
        
    
    def make_query(self, query: str):
        query = f"""{self.prompt}

Generate an outline for the given query.
{self.query_prefix}{query}
{self.outline_prefix}"""
        return query

    def __call__(self, llm, query: str, **kwargs) -> str:
        generation_query = self.make_query(query)
        response = llm.respond(generation_query, **kwargs)
        return response
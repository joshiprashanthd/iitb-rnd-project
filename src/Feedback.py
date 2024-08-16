from .utils import Question, gen_points_text, extract_feedback, extract_problems, extract_response

class Feedback:
    def __init__(self, question: Question) -> None:
        self.question = question
        self.gen_prompt_from_examples()
        
    def gen_prompt_from_examples(self):
        TEMPLATE = """Role: {role}
Goal: {goal}

Definitions:
{definitions}

Follow the below instructions carefully:
{emphases}

Follow the below instructions to generate feedback:
{feedback_instructions}

Examples:
{examples}
"""
        
        EXAMPLE_TEMPLATE = """Example {num}:{query_prefix}{query}
Text: {text}

{cot}
Response: {response}
Feedback: {feedback}
""" # this adds new line in the template
        examples = []
        for i, example in enumerate(self.question.examples):
            examples.append(EXAMPLE_TEMPLATE.format(num=i+1,
                                                query_prefix="\nQuery: " if self.question.add_query else "",
                                                query=example.query if self.question.add_query else "",
                                                text=example.text,
                                                cot=example.cot,
                                                response=example.response,
                                                feedback=example.feedback if example.feedback and len(example.feedback) > 0 else "None"))
        examples_text = "\n".join(examples)
        
        DEFINITION_TEMPLATE = """{term}: {definition}"""
        definitions = []
        for i, d in enumerate(self.question.definitions):
            definitions.append(DEFINITION_TEMPLATE.format(term=d.term, definition=d.definition))
        definitions_text = gen_points_text(definitions)
        
        
        feedback_instructions_text = gen_points_text(self.question.feedback_instructions)
        emphases_text = gen_points_text(self.question.emphases)
        
        self.prompt = TEMPLATE.format(role=self.question.role, 
                                      goal=self.question.goal, 
                                      definitions=definitions_text, 
                                      emphases=emphases_text,
                                      feedback_instructions=feedback_instructions_text,
                                      examples=examples_text)
    
    def make_query(self, query: str, text: str):
        
        prompt = """{prompt}Prompt:{query_prefix}{query}
Text: {text}
Problems: """
        return prompt.format(prompt=self.prompt,
                        query_prefix="\nQuery: " if self.question.add_query else "",
                        query=query if self.question.add_query else "",
                        text=text)

    def __call__(self, llm, query: str, text: str) -> str:
        generation_query = self.make_query(query, text)
        llm_response: str = llm.respond(generation_query)
        llm_response = llm_response.strip()
        
        response = extract_response(llm_response)
        if response == 'yes': return ""
        
        problems = extract_problems(llm_response)
        feedback = extract_feedback(llm_response)
        
        return f"{problems} {feedback}"
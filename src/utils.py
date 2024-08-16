import json, re, csv, datetime
from dataclasses import dataclass
from typing import Optional, List
import random

def extract_feedback(text: str):
    regex = r".*[Ff]eedback[\s]*:"
    after_regex = r".*\n*"
    text = re.sub(regex, "", text, flags=re.DOTALL).strip()
    matches = re.search(after_regex, text, re.MULTILINE)
    if matches:
        return matches.group().strip()
    return ""

def extract_response(text: str):
    regex = r".*Response:"
    after_regex = r".*\n*"

    if "Response:" not in text: return "yes"    
    
    text = re.sub(regex, "", text, flags=re.DOTALL).strip().lower()
    matches = re.search(after_regex, text, re.MULTILINE)
    
    if matches:
        result = matches.group().strip()
        if "no" in result: return "no"
        
    return "yes"

def extract_problems(text: str):
    after_regex = r".*\n*"
    matches = re.search(after_regex, text, re.MULTILINE)
    if matches:
        return matches.group().strip()
    return ""

def extract_multiline_points(text: str):
    points = []
    pattern = r'\d+\.(.*)|-(.*)'
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        for match in matches:
            points.append(match[0].strip())
    else:
        points.extend(text.split("\n"))
    return points

def extract_points(text: str):
    points = []
    for line in text.split("\n"):
        pattern = r'^[\s\-\d\W*]+\s*'
        line = re.sub(pattern, '', line)
        if len(line) > 0:
            points.append(line)
    return points

def gen_points_text(points, point_sep = "\n", numbered=False, default_prefix="- "):
    POINT_TEMP = """{point_prefix}{point}"""
    res = []
    for i, point in enumerate(points):
        res.append(POINT_TEMP.format(point_prefix=default_prefix if not numbered else f'{i+1}. ', point=point))
    return point_sep.join(res)

def clean_strings(strings):
    def clean(s: str):
        return s.strip().capitalize()
    strings = map(clean, strings)
    
    def addDot(s: str):
        if len(s) > 0 and s[-1] != '.':
            return s + '.'
        return s
    return map(addDot, strings)


def generate_time_string():
    now = datetime.datetime.now()
    return f"{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"

def get_queries(csv_path: str, max_len=None, random_seed=42, random_sampling=True):
    queries = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        queries = [(str(row[0]), row[1]) for row in reader if reader.line_num > 1]
        
    if not max_len: max_len = len(queries)
    
    if random_sampling and max_len < len(queries):
        random.seed(random_seed)
        queries = random.sample(queries, max_len)
        
    return queries

        
@dataclass
class ExampleData:
    query: str
    text: str
    cot: str
    response: str
    feedback: str
    refined: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            query = data.get('query'),
            text = data.get('text'),
            cot = data.get('cot'),
            response = data.get('response'),
            feedback = data.get('feedback'),
            refined = data.get('refined'),
        )
        
@dataclass
class DefinitionData:
    term: str
    definition: str
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            term = data.get('term'),
            definition = data.get('definition'),
        )
    
class Question:
    def __init__(self, json_path: str) -> None:
        self.role: str = ""
        self.goal: str = ""
        self.definitions: List[DefinitionData] = []
        self.emphases: List[str] = []
        self.feedback_instructions: List[str] = []
        self.examples: List[ExampleData] = []
        self.add_query: bool = False
        
        self._init_data(json_path)
        
    def _init_data(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(fp=f)
            self.role = data['role']
            self.goal = data['goal']
            self.emphases = data['emphases']
            self.add_query = data.get('add_query', False)
            self.feedback_instructions = data['feedback_instructions']
            
            for e in data['examples']:
                self.examples.append(ExampleData.from_dict(e))
            
            for d in data['definitions']:
                self.definitions.append(DefinitionData.from_dict(d))
            
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate


with open("src/backend/prompts/prompt.txt", 'r') as file:
    _template = file.read()
MY_PROMPT = ChatPromptTemplate.from_template(_template)

with open("src/backend/prompts/guidelines.txt", 'r') as file:
    GUIDELINES = file.read()

with open("src/backend/prompts/examples.txt", 'r') as file:
    EXAMPLES = file.read()


# SQL Database Query With NLP using LangChain Framework

## Overview
This Colab notebook provides a comprehensive guide on leveraging the LangChain framework to perform Natural Language Processing (NLP)-based queries on an employee database. The process involves importing essential Python packages, setting up embeddings, creating a vector database, and utilizing LangChain tools for semantic similarity and few-shot learning.

## Installation
```python
!pip install -r requirements.txt
```

## Setup
```python
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector

google_api_key = 'YourAPIKey'
```

## Data Preparation
```python
# Initializes GooglePalmEmbeddings with your API key.
embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)

# Loads data from a CSV file using CSVLoader, extracting specified metadata columns.
loader = CSVLoader(file_path='Data_to_vector.csv', metadata_columns=['sex', 'company', 'employement_type', 'department'])
data = loader.load()

# Creates key vectors using FAISS based on the loaded data and GooglePalmEmbeddings.
keyvectors = FAISS.from_documents(documents=data, embedding=embeddings)

# Sets up a SemanticSimilarityExampleSelector for the key vectors.
key_selector = SemanticSimilarityExampleSelector(vectorstore=keyvectors, k=2)
```

## Few-Shot Learning
```python
# Load few shots from file and convert them into vectors using FAISS
with open('few_shots.txt', 'r') as file:
    few_shots = eval(file.read())
few_content = [" ".join(example.values()) for example in few_shots]
few_shots_vector = FAISS.from_texts(few_content, embeddings, metadatas=few_shots)

# Set up SemanticSimilarityExampleSelector for few shots
few_shots_selector = SemanticSimilarityExampleSelector(vectorstore=few_shots_vector, k=2)
```

## Model Configuration
```python
# Initializes LLM with your API key.
llm = GooglePalm(google_api_key=google_api_key, temperature=0.2)

# Set up SQLDatabase connection
db = SQLDatabase.from_uri("sqlite:///employee_details.db")
```

## Prompt Creation
```python
# Creating the prompt with samples to give comments of LLM
example_prompt = PromptTemplate(
    input_variables=["company", "department", "employement_type", "sex", "Question", "SQLQuery", "SQLResult", "Answer"],
    template='You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question. ...')
```

## Combined Example Selector
```python
# In Langchain multiple example selector doesn't support, so create class to process multiple selector
class CombinedExampleSelector(BaseExampleSelector):
    def __init__(self, example_selectors):
        self.example_selectors = example_selectors

    def select_examples(self, input_variables):
        selected_examples = []
        for example_selector in self.example_selectors:
            examples = example_selector.select_examples(input_variables)
            selected_examples.append(examples)
        selected_examples = [{**dict1, **dict2} for dict1, dict2 in zip(selected_examples[0], selected_examples[1])]
        return selected_examples

    def add_example(self, example):
        for example_selector in self.example_selectors:
            example_selector.add_example(example)

combined_selector = CombinedExampleSelector([key_selector, few_shots_selector])
```

## Few-Shot Prompt Template
```python
# Set up FewShotPromptTemplate with a combined example selector, example prompt, and specified input variables.
few_shot_prompt = FewShotPromptTemplate(
    example_selector=combined_selector,
    example_prompt=example_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info"],  # These variables are used in the prefix and suffix
)
```

## Model Initialization
```python
# Initialise the chain model to query db
model = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

model.run('How many female employees have grater than 30 age in tesla and in AI department?')
```

## Now you are ready to query the database using natural language!
```python
model.run('How many female employees have grater than 30 age in tesla and in AI department?')

model.run('What are the departments in tesla?')

model.run('Which employee got higher bonus give the name?')
```
#### By Renukumar
Linkedin : https://www.linkedin.com/in/renukumar-r/

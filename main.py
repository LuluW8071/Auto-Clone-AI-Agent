from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import os
import ast

# Load environment variables from a .env file
load_dotenv()

# Initialize language model for general querying
llm = Ollama(model="llama3", request_timeout=30.0)

# Initialize document parser to process .pdf files
parser = LlamaParse(result_type="markdown")

# Setup file extractor for processing documents
file_extractor = {".pdf": parser}

# Load documents from the specified directory and parse them
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Resolve the embedding model and create a vector index from the documents
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create a query engine from the vector index and LLM
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools for querying and code reading
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="Provides documentation about code for an API. Use this for reading docs for the API.",
        ),
    ),
    code_reader,  # ---- `code_reader.py`
]

# Initialize a separate LLM for code generation
code_llm = Ollama(model="codellama")

# Create the ReActAgent with tools and the code generation LLM
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define the output data model and setup output parser
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

# Create a Pydantic parser for the CodeOutput model
parser = PydanticOutputParser(CodeOutput)

# Format the prompt template for code parsing
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)

# Setup a query pipeline with prompt template and LLM
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Main loop for querying the agent
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    # Retry loop for handling potential errors
    while retries < 3:
        try:
            # Query the agent with the user input
            result = agent.query(prompt)
            # Process the result through the output pipeline
            next_result = output_pipeline.run(response=result)
            # Clean and parse the output
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            # Handle exceptions and retry if necessary
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)

    # If retries are exhausted, notify user
    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    # Output the generated code and description
    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    # Attempt to save the generated code to a file
    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except Exception as e:
        print("Error saving file:", e)

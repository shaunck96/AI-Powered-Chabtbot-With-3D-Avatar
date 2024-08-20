from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# Ensure all necessary imports are included

# Initialize your system with necessary components
system_prompt = "You are a helpful assistant that extracts key search terms from user queries for vector database searches."
instruction = "Extract and list all relevant search terms from the user's query that should be used for a vector database search."
user_input = "What are some recent developments in the field of vision AI?"

# Define a new response schema for extracting search terms
response_schemas = [
    ResponseSchema(
        name="List of keyword search terms for vector database",
        description="Extract relevant keywords from the user's query for vector database searches.",
        format_instructions="List all relevant keywords, separated by commas."
    )
]

# Initialize the output parser with the new response schema
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Construct the messages to be sent to the model
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"{instruction} User Query: {user_input} Format Instructions: {format_instructions}")
]

# Initialize the chat model (ensure you have the correct API key and parameters)
chat_model = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))  # Replace with your method of fetching the API key
response = chat_model.generate_response(messages)

# The response from the chat model can now be parsed for vector search terms
extracted_terms = response.get("content")  # You might need to adjust this depending on the response format
print(extracted_terms)  # This will print out the extracted search terms

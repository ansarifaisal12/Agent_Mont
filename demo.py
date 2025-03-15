# app.py

import google.generativeai as genai
from agent_mont import AgentMontExtended
from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Initialize Agent Mont monitoring with Gemini model
mont = AgentMontExtended(model="gemini-1.5-flash")

# Start monitoring
mont.start()

# Example: Use a tool to scrape a website
scrape_tool = ScrapeWebsiteTool(website_url='https://en.wikipedia.org/wiki/Artificial_intelligence')
text = scrape_tool.run()
print("Scraped Text:", text[:500], "...")  # Print a preview

# Write the scraped text to a file
file_writer_tool = FileWriterTool()
text_cleaned = text.encode("ascii", "ignore").decode()
write_result = file_writer_tool._run(filename='ai.txt', content=text_cleaned, overwrite="True")
print("File Write Result:", write_result)

# Search for a specific context in the text
txt_search_tool = TXTSearchTool(txt='ai.txt')
context = txt_search_tool.run('What is natural language processing?')
print("Context for NLP:", context)

# Define Gemini Function for Processing Text
def use_gemini(prompt):
    """ Uses Google Gemini to generate responses based on context. """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Create a Crew AI agent using Gemini API
data_analyst = Agent(
    role='Educator',
    goal=f'Based on the provided context, explain "What is Natural Language Processing?"\n\nContext: {context}',
    backstory='You are an AI-powered data educator.',
    verbose=True,
    allow_delegation=False,
    tools=[txt_search_tool]
)

# Define a task for the agent
test_task = Task(
    description="Understand the topic and generate an accurate response",
    tools=[txt_search_tool],
    agent=data_analyst,
    expected_output='Provide a correct and detailed explanation of NLP.'
)

# Create a Crew and start the task
crew = Crew(
    agents=[data_analyst],
    tasks=[test_task]
)

output = crew.kickoff()
print("Crew Output:", output)

# Set token usage based on Crew AI output
mont.set_token_usage_from_crew_output(output)

# End monitoring and log performance
mont.end()

# Visualize results (CLI or Streamlit dashboard)
mont.visualize(method='cli')

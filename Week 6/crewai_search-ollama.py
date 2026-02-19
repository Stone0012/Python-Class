"""
Minimal Search Engine using CrewAI and Ollama

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull llama2
3. Install dependencies: pip install crewai crewai-tools langchain-community
"""

from crewai import Agent, Task, Crew, Process, LLM
import json

# Initialize Ollama LLM using CrewAI's LLM wrapper
llm = LLM(model="ollama/llama3.2", base_url="http://lambda2.uncw.edu:11434/api/generate")

# Define search results database
SEARCH_DATABASE = {
    "python": "Python is a high-level programming language known for its simplicity and readability.",
    "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks requiring human intelligence.",
    "crewai": "CrewAI is a framework for orchestrating autonomous AI agents to work together on complex tasks.",
    "ollama": "Ollama is a tool for running large language models locally on your machine."
}

def perform_search(query: str) -> str:
    """
    Simple search function that doesn't require tool calling.
    """
    for keyword, info in SEARCH_DATABASE.items():
        if keyword.lower() in query.lower():
            return f"Search result for '{query}': {info}"
    
    return f"No specific results found for '{query}'. Try searching for: python, ai, crewai, or ollama."

# Create the search agent (without tools to avoid function calling issues)
search_agent = Agent(
    role='Search Specialist',
    goal='Find and retrieve relevant information based on user queries by performing searches',
    backstory='An expert at finding and synthesizing information from various sources.',
    llm=llm,
    verbose=True
)

# Create the synthesizer agent
synthesizer_agent = Agent(
    role='Information Synthesizer',
    goal='Analyze and summarize search results into clear, concise answers',
    backstory='A skilled analyst who can distill complex information into easy-to-understand summaries.',
    llm=llm,
    verbose=True
)

def search(query: str) -> str:
    """
    Main search function that orchestrates the agents.
    
    Args:
        query: The search query from the user
        
    Returns:
        A synthesized answer based on search results
    """
    # Perform the search directly
    search_result = perform_search(query)
    
    # Define tasks
    search_task = Task(
        description=f"Based on this search result: '{search_result}', provide a comprehensive answer to: {query}",
        agent=search_agent,
        expected_output="A helpful answer based on the search result"
    )
    
    synthesis_task = Task(
        description=f"Take the previous answer and provide an even more detailed, well-organized summary answering: {query}",
        agent=synthesizer_agent,
        expected_output="A comprehensive, well-structured summary"
    )
    
    # Create crew
    crew = Crew(
        agents=[search_agent, synthesizer_agent],
        tasks=[search_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute the search
    result = crew.kickoff()
    return result

# Main execution
if __name__ == "__main__":
    print("=== Minimal Search Engine with CrewAI and Ollama ===\n")
    
    # Example searches
    queries = [
        "What is Python?",
        "Tell me about AI",
        "What is CrewAI?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        try:
            result = search(query)
            print(f"\nFinal Answer:\n{result}\n")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        user_query = input("Enter your search query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            result = search(user_query)
            print(f"\nAnswer:\n{result}\n")
        except Exception as e:
            print(f"Error: {e}\n")
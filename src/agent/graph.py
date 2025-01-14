import asyncio
from typing import cast, Any, Literal
import json

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Search

tavily_async_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

## READ: OK
def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration // READ: OK
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries // READ: OK
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions - parses the user input based on the schema
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company, # From input
        info=json.dumps(state.extraction_schema, indent=2), # From input, optional with default schema provided
        user_notes=state.user_notes, # Optional field
        max_search_queries=max_search_queries, # From configuration
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries] # LLM in results will return a list of queries
    return {"search_queries": query_list} # Outputs the list of queries, function returns a dict of "search_queries" key and the list of queries as the value


async def research_company(
    state: OverallState, config: RunnableConfig # Runnable config determines max search results, OverallState contains search queries
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """

    # Get configuration // READ: OK
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results # Max search results based on configuration in runnable config

    # Search tasks // READ: OK
    search_tasks = []
    for query in state.search_queries: # Iterates over the list of search queries that were generated in the generate_queries function, stored in OverallState
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results, # With each query, the max search results is set to the configurable max search results
                include_raw_content=True, # Include raw content from the search results
                topic="general", # Topic is set to general, other options are "company", "person", "product", "technology", "industry", "location", "event", "news", "legal", "financial", "technology", "product", "person", "company", "event", "location", "industry", "news", "legal", "financial"
            )
        )

    # Execute all searches concurrently // READ: OK
    search_docs = await asyncio.gather(*search_tasks) # Executes all searches concurrently

    # Deduplicate and format sources // READ: OK
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    ) # Uses the deduplicate_and_format_sources function in utils.py to deduplicate and format the search results

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), # Schema from input
        content=source_str, # Raw content from the search results
        company=state.company, # Company from input
        user_notes=state.user_notes, # User notes from input, Optional field
    )
    result = await claude_3_5_sonnet.ainvoke(p) # Asynchronously invokes the LLM with the prompt: INFO_PROMPT
    # Returns a dict of "completed_notes" key; value is a list with a single string - the output of LLM generated with the prompt: INFO_PROMPT
    return {"completed_notes": [str(result.content)]} 
    


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result} # Returns a dict of "info" key; value is the output of the structured LLM generated with the prompt: EXTRACTION_PROMPT


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput) # Requires structured output based on ReflectionOutput

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2), # Schema from input
        info=state.info, 
        # info generated from gather_notes_extract_schema based on the prompt: EXTRACTION_PROMPT, which extracts the information from the raw content
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt}, 
                # System prompt from REFLECTION_PROMPT, which reviews the extracted information against the schema
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries, # this will return a different list of search queries based on the missing information
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company" # when this is returned, the graph will execute the research_company node

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection) 
# END gets executed in route_from_reflection, when either the reflection is satisfactory or the max reflection steps is reached

# Compile
graph = builder.compile()

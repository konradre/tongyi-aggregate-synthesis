# Usage Examples

## Basic Search

Search across all configured connectors:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kubernetes best practices 2024",
    "top_k": 10
  }'
```

## Selective Connector Search

Use only specific connectors:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning frameworks comparison",
    "top_k": 5,
    "connectors": ["searxng"]
  }'
```

## Research with Synthesis

Full research with LLM-powered synthesis and citations:

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key differences between RAG and fine-tuning for LLMs?",
    "top_k": 10,
    "reasoning_effort": "high"
  }'
```

Response includes synthesized answer with inline citations:

```json
{
  "content": "RAG (Retrieval Augmented Generation) and fine-tuning are two distinct approaches to customizing LLMs [sx_a1b2c3d4].\n\n## RAG\n\nRAG works by retrieving relevant documents at inference time and including them in the context [tv_e5f6g7h8]. Key advantages:\n- No model retraining required\n- Easy to update knowledge base\n- Lower computational cost\n\n## Fine-tuning\n\nFine-tuning modifies the model weights through additional training [sx_c9d0e1f2]...",
  "citations": [
    {"id": "sx_a1b2c3d4", "title": "RAG vs Fine-tuning Guide", "url": "https://..."},
    {"id": "tv_e5f6g7h8", "title": "Understanding RAG", "url": "https://..."}
  ]
}
```

## Quick Answers

For fast, concise answers:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the latest stable Python version?"
  }'
```

## Python Client

```python
import httpx

async def search(query: str, top_k: int = 10):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/search",
            json={"query": query, "top_k": top_k}
        )
        return response.json()

async def research(query: str, reasoning_effort: str = "medium"):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/research",
            json={
                "query": query,
                "reasoning_effort": reasoning_effort
            }
        )
        return response.json()

# Usage
import asyncio

async def main():
    # Quick search
    results = await search("python async tutorial")
    for source in results["sources"][:3]:
        print(f"- {source['title']}: {source['url']}")

    # Deep research
    research_result = await research(
        "How do async context managers work in Python?",
        reasoning_effort="high"
    )
    print(research_result["content"])

asyncio.run(main())
```

## JavaScript/TypeScript Client

```typescript
interface Source {
  id: string;
  title: string;
  url: string;
  content: string;
  score: number;
  connector: string;
}

interface ResearchResponse {
  query: string;
  content: string;
  citations: { id: string; title: string; url: string }[];
  sources: Source[];
}

async function research(query: string): Promise<ResearchResponse> {
  const response = await fetch("http://localhost:8000/api/v1/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: 10,
      reasoning_effort: "medium"
    })
  });
  return response.json();
}

// Usage
const result = await research("What is WebAssembly?");
console.log(result.content);
```

## Integration with LangChain

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import httpx

class ResearchInput(BaseModel):
    query: str = Field(description="Research query")

class ResearchTool(BaseTool):
    name = "research"
    description = "Research a topic using multi-source search and LLM synthesis"
    args_schema = ResearchInput

    def _run(self, query: str) -> str:
        response = httpx.post(
            "http://localhost:8000/api/v1/research",
            json={"query": query, "reasoning_effort": "medium"},
            timeout=120.0
        )
        result = response.json()
        return result["content"]

    async def _arun(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:8000/api/v1/research",
                json={"query": query, "reasoning_effort": "medium"}
            )
            result = response.json()
            return result["content"]

# Use in LangChain agent
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
tools = [ResearchTool()]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

result = agent.run("Research the latest developments in quantum computing")
```

## MCP Server Wrapper

Create an MCP server to expose the research tool:

```python
# mcp_server.py
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

server = Server("research-tool")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="research",
            description="Research a topic with multi-source search and synthesis",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                    "reasoning_effort": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search",
            description="Multi-source web search without synthesis",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    async with httpx.AsyncClient(timeout=120.0) as client:
        if name == "research":
            response = await client.post(
                "http://localhost:8000/api/v1/research",
                json=arguments
            )
            result = response.json()
            return [TextContent(type="text", text=result["content"])]

        elif name == "search":
            response = await client.post(
                "http://localhost:8000/api/v1/search",
                json=arguments
            )
            result = response.json()
            sources_text = "\n".join([
                f"[{s['id']}] {s['title']}\n{s['url']}\n{s['content'][:200]}..."
                for s in result["sources"]
            ])
            return [TextContent(type="text", text=sources_text)]

if __name__ == "__main__":
    asyncio.run(server.run())
```

## Batch Research

Process multiple queries:

```python
import asyncio
import httpx

async def batch_research(queries: list[str]):
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            client.post(
                "http://localhost:8000/api/v1/research",
                json={"query": q, "reasoning_effort": "low"}
            )
            for q in queries
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

queries = [
    "What is Docker?",
    "What is Kubernetes?",
    "What is Terraform?"
]

results = asyncio.run(batch_research(queries))
for result in results:
    print(f"## {result['query']}\n")
    print(result["content"][:500])
    print("\n---\n")
```

## Perplexity API Migration

Replace Perplexity API calls:

```python
# Before (Perplexity)
from openai import OpenAI

client = OpenAI(
    api_key="pplx-xxx",
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar-deep-research",
    messages=[{"role": "user", "content": "What is RAG?"}]
)

# After (Research Tool)
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/research",
    json={
        "query": "What is RAG?",
        "reasoning_effort": "high"  # Similar to sonar-deep-research
    },
    timeout=120.0
)
result = response.json()
print(result["content"])
```

The research tool provides similar functionality:
- Multi-source search (like Perplexity's web search)
- Synthesized answers with citations
- Configurable reasoning depth

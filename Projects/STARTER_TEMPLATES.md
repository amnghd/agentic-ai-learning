# Starter Code Templates

## Week 1: Environment Setup

### Dockerfile Template
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml Template
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## Week 2: SLM Fine-Tuning

### Fine-Tuning Script Template
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def setup_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def setup_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

def train_model(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model

# Usage
if __name__ == "__main__":
    model, tokenizer = setup_model_and_tokenizer("meta-llama/Llama-2-7b-hf")
    model = setup_lora(model)
    dataset = load_dataset("your-dataset")
    trained_model = train_model(model, tokenizer, dataset)
```

## Week 3: RAG System

### RAG Pipeline Template
```python
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def setup_rag_pipeline(index_name, documents):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Pinecone.from_documents(
        documents,
        embeddings,
        index_name=index_name
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def query_rag(qa_chain, question):
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }
```

## Week 4: AutoGen

### Multi-Agent Template
```python
import autogen

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key",
    }
]

# Create agents
triage_agent = autogen.AssistantAgent(
    name="triage_agent",
    system_message="You are a triage agent. Route queries to appropriate specialists.",
    llm_config={"config_list": config_list}
)

support_agent = autogen.AssistantAgent(
    name="support_agent",
    system_message="You are a customer support agent. Help customers with their questions.",
    llm_config={"config_list": config_list}
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[triage_agent, support_agent],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# Start conversation
def handle_customer_query(query):
    response = manager.initiate_chat(
        message=query,
        recipient=triage_agent
    )
    return response
```

## Week 5: LangGraph

### State Machine Template
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    classification: str
    response: str
    confidence: float

def classify_query(state: State) -> State:
    # Classification logic
    state["classification"] = "technical"
    return state

def route_query(state: State) -> State:
    # Routing logic
    if state["classification"] == "technical":
        state["response"] = "Route to technical support"
    return state

def process_query(state: State) -> State:
    # Processing logic
    state["response"] = "Here's your answer..."
    state["confidence"] = 0.95
    return state

# Build graph
workflow = StateGraph(State)
workflow.add_node("classify", classify_query)
workflow.add_node("route", route_query)
workflow.add_node("process", process_query)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "route")
workflow.add_edge("route", "process")
workflow.add_edge("process", END)

app = workflow.compile()
```

## Week 6: Braintrust

### Evaluation Template
```python
import braintrust

def evaluate_agent(query, expected_answer):
    # Run agent
    actual_answer = agent.query(query)
    
    # Evaluate
    return {
        "input": query,
        "expected": expected_answer,
        "actual": actual_answer,
        "metrics": {
            "accuracy": calculate_accuracy(actual_answer, expected_answer),
            "latency": measure_latency(),
            "cost": calculate_cost()
        }
    }

# Run evaluation
results = braintrust.run(
    name="agent-evaluation",
    experiment=evaluate_agent,
    data=[
        {"query": "What are your hours?", "expected": "We're open 9-5"},
        # ... more test cases
    ]
)
```

## Week 7: API

### FastAPI Template
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    user_id: str

class QueryResponse(BaseModel):
    response: str
    sources: list
    confidence: float

@app.post("/api/v1/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # Process query through agentic system
        result = agentic_system.process(request.query)
        
        return QueryResponse(
            response=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## Week 8: Integration

### Main Orchestration Template
```python
class AgenticCustomerSupport:
    def __init__(self):
        self.rag_system = setup_rag()
        self.finetuned_model = load_model()
        self.autogen_agents = setup_autogen()
        self.langgraph_workflow = setup_langgraph()
    
    def process_query(self, query: str, user_id: str):
        # 1. Try RAG first
        rag_result = self.rag_system.query(query)
        
        if rag_result["confidence"] > 0.8:
            return rag_result
        
        # 2. Use fine-tuned model
        model_result = self.finetuned_model.generate(query)
        
        if model_result["confidence"] > 0.7:
            return model_result
        
        # 3. Route to agents
        agent_result = self.autogen_agents.handle(query)
        
        # 4. Process through workflow
        final_result = self.langgraph_workflow.invoke({
            "query": query,
            "initial_response": agent_result
        })
        
        return final_result
```

## Usage Notes

1. Replace placeholder values with actual configuration
2. Add error handling
3. Implement logging
4. Add tests
5. Follow security best practices

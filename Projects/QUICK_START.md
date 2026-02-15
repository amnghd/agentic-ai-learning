# Quick Start Guide

## Prerequisites

Before starting, ensure you have:
- Python 3.11 or higher
- Docker and Docker Compose
- Git
- Access to cloud resources (optional, for deployment)

## Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd agentic-customer-support
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 5. Start Docker Services
```bash
docker-compose up -d
```

### 6. Run Tests
```bash
pytest tests/
```

### 7. Start Development Server
```bash
uvicorn src.api.main:app --reload
```

## Week-by-Week Setup

### Week 1: Environment Setup
Follow the setup instructions above. Complete Project 1.1.

### Week 2: SLM Fine-Tuning
1. Set up Hugging Face account
2. Get access to model (Llama-2, Mistral, etc.)
3. Prepare dataset
4. Start fine-tuning

### Week 3: RAG System
1. Choose vector database (Pinecone, Weaviate, etc.)
2. Set up embeddings
3. Ingest documents
4. Test retrieval

### Week 4: AutoGen
1. Install AutoGen: `pip install pyautogen`
2. Set up LLM provider (OpenAI, Anthropic)
3. Create first agent
4. Build multi-agent system

### Week 5: LangGraph & LangSmith
1. Install LangGraph: `pip install langgraph`
2. Set up LangSmith account
3. Create state machine
4. Enable tracing

### Week 6: Braintrust
1. Install Braintrust: `pip install braintrust`
2. Set up Braintrust account
3. Create test cases
4. Run evaluations

### Week 7: Integration
1. Integrate all components
2. Create API
3. Set up database
4. Test end-to-end

### Week 8: Capstone
1. Complete integration
2. Deploy to production
3. Write documentation
4. Prepare presentation

## Common Issues

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size, use gradient checkpointing, or use smaller model

### Issue: API Rate Limits
**Solution**: Implement rate limiting and retry logic

### Issue: Docker Issues
**Solution**: Ensure Docker is running and has enough resources allocated

### Issue: Import Errors
**Solution**: Check virtual environment is activated and dependencies are installed

## Getting Help

1. Check documentation in `docs/` folder
2. Review week-specific README files
3. Check framework documentation
4. Ask in community forums
5. Review example code

## Next Steps

1. Complete Week 1 setup
2. Review Week 1 project requirements
3. Start implementing Week 1 projects
4. Join community for support

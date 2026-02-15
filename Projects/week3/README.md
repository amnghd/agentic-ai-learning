# Week 3: Retrieval Augmented Generation (RAG)

## Project 3.1: Vector Database & Embeddings

### Objectives
- Set up vector database
- Implement embedding pipeline
- Create document ingestion system
- Design chunking strategies

### Tasks

#### Task 1: Vector Database Setup
Choose one:
- **Pinecone**: Managed, easy to use
- **Weaviate**: Open-source, feature-rich
- **Qdrant**: High performance
- **Chroma**: Simple, local-first

#### Task 2: Embedding Model Selection
- Compare embedding models:
  - OpenAI text-embedding-ada-002
  - Cohere embed-english-v3.0
  - Sentence Transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- Fine-tune embeddings on customer support data (optional)

#### Task 3: Document Ingestion Pipeline
- Load documents (PDFs, markdown, text files)
- Extract text
- Clean and normalize
- Chunk documents:
  - Fixed-size chunks
  - Sentence-aware chunking
  - Semantic chunking
- Generate embeddings
- Store in vector database

#### Task 4: Chunking Strategies
Implement multiple strategies:
- **Fixed-size**: 512 tokens with overlap
- **Recursive**: By paragraphs/sentences
- **Semantic**: Using embeddings to find boundaries

### Deliverables
- Vector database with indexed documents
- Document ingestion pipeline
- Embedding generation code
- Chunking strategy comparison

---

## Project 3.2: Advanced RAG Patterns

### Objectives
- Implement advanced retrieval techniques
- Build query understanding
- Create reranking system
- Implement hybrid search

### Tasks

#### Task 1: Multi-Query Retrieval
- Generate multiple query variations
- Retrieve from each query
- Combine and deduplicate results

#### Task 2: Reranking
- Implement cross-encoder reranking
- Use models like:
  - ms-marco-MiniLM-L-6-v2
  - bge-reranker-base
- Rerank top-K results

#### Task 3: Hybrid Search
- Combine semantic and keyword search
- Use BM25 for keyword search
- Weighted combination of results
- Implement reciprocal rank fusion

#### Task 4: Query Expansion
- Generate query variations
- Extract entities
- Add synonyms
- Context-aware expansion

#### Task 5: Contextual Compression
- Compress retrieved context
- Remove irrelevant information
- Extract key facts
- Maintain coherence

### Deliverables
- Advanced RAG pipeline
- Comparison of retrieval strategies
- Performance benchmarks
- Code implementation

### Evaluation Metrics
- Retrieval accuracy (MRR, NDCG)
- Answer quality
- Latency
- Cost per query

---

## Project 3.3: RAG Evaluation & Monitoring

### Objectives
- Build evaluation framework
- Set up monitoring
- Create feedback loops
- Implement A/B testing

### Tasks

#### Task 1: Evaluation Framework
Use RAGAS or TruLens:
- **Faithfulness**: Answer grounded in context?
- **Answer Relevance**: Answer relevant to question?
- **Context Precision**: Relevant context retrieved?
- **Context Recall**: All relevant context retrieved?

#### Task 2: Monitoring
- Track metrics:
  - Query latency
  - Retrieval quality
  - Generation quality
  - User feedback
- Set up alerts
- Create dashboards

#### Task 3: Feedback Loop
- Collect user feedback
- Store feedback in database
- Use feedback to improve:
  - Retrieval
  - Chunking
  - Embeddings

#### Task 4: A/B Testing
- Set up A/B testing framework
- Test different:
  - Chunking strategies
  - Embedding models
  - Retrieval methods
- Statistical significance testing

### Deliverables
- Evaluation framework
- Monitoring dashboard
- Feedback collection system
- A/B testing infrastructure

### Resources
- [RAGAS Documentation](https://docs.ragas.io/)
- [TruLens Documentation](https://www.truera.com/trulens-eval/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)

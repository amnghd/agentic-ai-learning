# Week 5: Agentic Frameworks - LangGraph & LangSmith

## Project 5.1: State Machines with LangGraph

### Objectives
- Build customer support workflow as state machine
- Implement conditional routing
- Add human-in-the-loop
- Create error recovery

### State Machine Design

#### States
1. **Initial**: Receive customer query
2. **Classification**: Classify query type
3. **Routing**: Route to appropriate handler
4. **Processing**: Agent processes query
5. **Human Review**: Escalate to human (if needed)
6. **Resolution**: Provide solution
7. **Follow-up**: Check satisfaction
8. **Complete**: Close ticket

#### Edges (Transitions)
- Classification → Routing
- Routing → Processing
- Processing → Human Review (if confidence low)
- Processing → Resolution (if confidence high)
- Human Review → Resolution
- Resolution → Follow-up
- Follow-up → Complete

### Tasks

#### Task 1: Basic State Machine
- Install LangGraph
- Define state schema
- Create nodes (states)
- Define edges (transitions)
- Compile graph

#### Task 2: Conditional Routing
- Implement routing logic
- Add confidence thresholds
- Create decision nodes
- Handle multiple paths

#### Task 3: Human-in-the-Loop
- Add human review state
- Implement approval workflow
- Create notification system
- Handle human feedback

#### Task 4: Error Recovery
- Add error handling states
- Implement retry logic
- Create fallback paths
- Log errors

#### Task 5: State Persistence
- Save state to database
- Implement state recovery
- Add checkpointing
- Handle interruptions

### Deliverables
- LangGraph state machine
- Workflow documentation
- State persistence system
- Error handling

### Example Workflow
```
Customer Query
    ↓
Classification (Technical Issue)
    ↓
Routing (Technical Support)
    ↓
Processing (Agent generates response)
    ↓
Confidence Check (High)
    ↓
Resolution (Send response)
    ↓
Follow-up (Check satisfaction)
    ↓
Complete
```

---

## Project 5.2: Observability with LangSmith

### Objectives
- Integrate LangSmith for tracing
- Set up monitoring
- Create debugging tools
- Track costs and performance

### Tasks

#### Task 1: LangSmith Integration
- Set up LangSmith account
- Configure API keys
- Instrument LangChain/LangGraph code
- Enable tracing

#### Task 2: Tracing
- Trace all LLM calls
- Trace tool executions
- Trace agent decisions
- Trace state transitions

#### Task 3: Monitoring Dashboard
- Create custom dashboards
- Monitor:
  - Latency
  - Token usage
  - Costs
  - Error rates
  - Success rates

#### Task 4: Debugging Tools
- Use LangSmith playground
- Debug failed runs
- Analyze agent behavior
- Identify bottlenecks

#### Task 5: Cost Tracking
- Track costs per:
  - Query
  - Agent
  - Tool
  - Day/week/month
- Set up cost alerts
- Optimize expensive operations

#### Task 6: Custom Logging
- Add custom logs
- Log business events
- Track user interactions
- Store for analysis

### Deliverables
- LangSmith integration
- Monitoring dashboards
- Cost tracking system
- Debugging documentation

### Metrics to Track
- **Latency**: P50, P95, P99
- **Cost**: Per query, per day
- **Quality**: Success rate, user satisfaction
- **Errors**: Error rate, error types
- **Usage**: Queries per day, peak times

### Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangSmith Tracing Guide](https://docs.smith.langchain.com/tracing)

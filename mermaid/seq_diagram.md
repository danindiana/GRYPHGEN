<p align="center">
  <img src="https://github.com/user-attachments/assets/2ec238a0-b5ed-4000-afbc-1b27cc77cd64" alt="ChatGPT Image" width="400"/>
</p>


```
sequenceDiagram
    participant CG as CodeGenerator
    participant CA as CodeAnalyzer
    participant TM as TaskMonitor
    participant WO as WorkflowOptimizer
    participant ULS as UbuntuLinuxServer
    participant SYMORQ as SYMORQ
    participant SYMORG as SYMORG
    participant SYMAUG as SYMAUG

    CG->>SYMORQ: Publish "Code Generated" event
    SYMORQ->>SYMORG: Trigger LLM Automated RAG constructor
    SYMORG->>SYMAUG: Construct RAG for LLM
    SYMAUG->>SYMORQ: Acknowledge RAG construction

    loop SYMORQ sends instructions to ULS for resource allocation
        SYMORQ->>ULS: Request resource allocation
        ULS->>SYMORQ: Acknowledge resource allocation
    end

    loop SYMAUG starts containerized microservices on request
        SYMAUG->>ULS: Request to start containerized microservice
        ULS->>SYMAUG: Acknowledge to start containerized microservice
    end

    CG->>CA: Request code analysis
    CA->>TM: Subscribe to "code_generated" and "analysis_request" topics
    CA->>CG: Acknowledge code analysis request

    TM->>CA: Publish "Analysis Completed" event
    CA->>WO: Subscribe to "analysis_completed" and "task_execution_status" topics
    CA->>TM: Acknowledge analysis completed

    TM->>ULS: Request task execution status
    ULS->>TM: Publish "Task Execution Status" on "task_execution_status" topic
    TM->>ULS: Acknowledge task execution status request

    loop WO optimizes workflow based on analysis and task execution status
        WO->>CA: Publish "Workflow Optimization" message
        CA->>WO: Acknowledge workflow optimization message
    end

    ULS->>TM: Publish "Task Execution Status" on "task_execution_status" topic
    TM->>ULS: Acknowledge request for task execution status

    ULS->>TM: Publish "Execution Results" on "execution_results" topic
    TM->>ULS: Acknowledge execution results

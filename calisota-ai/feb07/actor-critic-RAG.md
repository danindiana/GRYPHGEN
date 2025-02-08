```mermaid
graph TD
    %% FAISS-Backed RAG System
    subgraph RAG_System ["Retrieval-Augmented Generation (RAG)"]
        RAG1["FAISS Vector DB"] 
        RAG2["Retriever & Embedder"]
        
        RAG2 -->|Query| RAG1
        RAG1 -->|Relevant Data| RAG2
    end

    %% Ensemble 1
    subgraph Ensemble_1 ["Ensemble 1"]
        A1["Large 'Slow-thinker' Language Model"] -->|Guidance/Inference| B1["Smaller 'Fast-thinker' Code Generator"]
        A1 -->|Feedback| C1["Smaller 'Fast-thinker' Actor-Critic"]
        
        B1 -->|Generates Code| D1["Execution Environment"]
        C1 -->|Evaluates Outputs| D1
        
        D1 -->|Results & Feedback| C1
        D1 -->|Refinement Requests| B1
        
        %% Integration with RAG
        A1 -->|Retrieves Context| RAG2
        B1 -->|Retrieves Code Samples| RAG2
    end
    
    %% Ensemble 2
    subgraph Ensemble_2 ["Ensemble 2"]
        A2["Large 'Slow-thinker' Language Model"] -->|Guidance/Inference| B2["Smaller 'Fast-thinker' Code Generator"]
        A2 -->|Feedback| C2["Smaller 'Fast-thinker' Actor-Critic"]
        
        B2 -->|Generates Code| D2["Execution Environment"]
        C2 -->|Evaluates Outputs| D2
        
        D2 -->|Results & Feedback| C2
        D2 -->|Refinement Requests| B2
        
        %% Integration with RAG
        A2 -->|Retrieves Context| RAG2
        B2 -->|Retrieves Code Samples| RAG2
    end
    
    %% Cross-Ensemble Communication
    D1 -->|Shares Insights| A2
    D2 -->|Shares Insights| A1
```

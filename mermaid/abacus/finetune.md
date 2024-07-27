```mermaid
graph TD
    MassiveDataset[Massive dataset] -->|Pre-training| PreTrainedLLM[Pre-trained LLM]
    DomainSpecificDataset[Domain-specific dataset] -->|Fine-tuning| FineTunedLLM[Fine-tuned LLM]
    PreTrainedLLM --> FineTunedLLM
    User[User] -->|Query| FineTunedLLM
    FineTunedLLM -->|Response| User
```

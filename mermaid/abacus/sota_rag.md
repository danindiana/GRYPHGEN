```mermaid
graph TD
    User[User] <-->|Conversation| LanguageModel[Language Model]
    LanguageModel -->|Send Data| RankingAlgorithms[Ranking Algorithms]
    RankingAlgorithms -->|Fetch Data| Index
    Index -->|Send Data| RankingAlgorithms
    Index -->|Parsing & Indexing| WebPages[Web Pages]
    WebPages -->|Crawl| Crawler
    Crawler -->|Send Data| ContextKeywords[Context and Keywords Extractor]
    ContextKeywords -->|Send Data| LanguageModel
    Crawler -->|Send Data| Index
    LanguageModel -->|Fetch Data| RankingAlgorithms
    RankingAlgorithms -->|Send Data| LanguageModel
```

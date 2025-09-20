```mermaid
flowchart LR
    Rel["Relational Context Rel(t)"] --> Attn["Attention Controller Φ"]
    Attn -->|"a(t)"| Gate["Edge Gating σ"]
    Gate -->|"W(e,t)"| V1["Node V1<br>RNN¹_Θ"]
    Gate -->|"W(e,t)"| LM["Node LM<br>RNN²_Θ"]
    Gate -->|"W(e,t)"| N3["Node n₃<br>RNNⁿ_Θ"]
    Gate -->|"W(e,t)"| N4["Node n₄<br>RNNⁿ_Θ"]

    Stim["External Stimulus Stim(t)"] --> V1
    Stim --> LM
    Stim --> N3
    Stim --> N4

    V1 -. "Weighted" .-> LM
    V1 -. "Weighted" .-> N3
    LM -. "Weighted" .-> V1
    LM -. "Weighted" .-> N4
    N3 -. "Weighted" .-> N4
    N4 -. "Weighted" .-> V1

    V1 --> Omega["Evidence Model Ω<br>P(cause|S_N;Rel)"]
    LM --> Omega
    N3 --> Omega
    N4 --> Omega

    Omega --> Post["Posterior B(t+1)"]
    Post --> Obj["Minimize KL[Q_t||P(⋅)]"]
    Post -. "t+1" .-> Rel

    classDef step1 fill:#4fc3f7,stroke:#fff,color:#fff;
    classDef step2 fill:#7e57c2,stroke:#fff,color:#fff;
    classDef step3 fill:#26a69a,stroke:#fff,color:#fff;
    classDef step4 fill:#ff7043,stroke:#fff,color:#fff;
    class Attn step1
    class Gate step2
    class V1,LM,N3,N4 step3
    class Omega,Post step4
```

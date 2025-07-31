```mermaid
graph TD
    classDef darkBox fill:#2d3436,color:white,stroke:#636e72,stroke-width:2px
    
    subgraph Experimental Setup
        A[Optogenetic Silencing] --> B[Varying Time Windows]
        B --> C[Simultaneous Recordings]
        C --> D[V1 & LM Cortical Areas]
        class A,B,C,D darkBox
    end

    subgraph Cortical Communication
        D --> E[Feedforward Pathway<br>V1 → LM]
        D --> F[Feedback Pathway<br>LM → V1]
        
        E --> G[Relatively Stable<br>Target-specific Subpopulations]
        F --> H[Highly Dynamic<br>Changes within 10s of ms]
        class E,F,G,H darkBox
    end

    subgraph Behavioral Modulation
        I[Reward Context] --> J[Enhanced Feedback Dynamics]
        J --> H
        class I,J darkBox
    end

    subgraph Population Activity Effects
        H --> K[Rotating Communication Dimensions]
        K --> L[Dynamic Functional Subnetworks]
        L --> M[V1 Covariance Structure Rotation]
        M --> N[Context-Dependent Processing]
        class K,L,M,N darkBox
    end

    subgraph Key Findings
        O[Communication Channels<br>Reorganize Dynamically]
        P[Feedback Modulates<br>V1 Processing Geometry]
        Q[Behavioral Relevance<br>Accelerates Dynamics]
        class O,P,Q darkBox
    end

    K --> O
    M --> P
    J --> Q
```

Source Paper: bioRxiv preprint doi: https://doi.org/10.1101/2021.06.28.449892; this version posted June 28, 2021. Dynamic causal communication channels
between neocortical areas Mitra Javadzadeh*, Sonja B. Hofer* Sainsbury Wellcome Centre for Neural Circuits and Behaviour, University College London, London,
United Kingdom

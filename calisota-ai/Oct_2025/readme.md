```mermaid
flowchart LR
  %% ===== Core Stacks =====
  subgraph HW[Hardware Stack - Calisota Engine]
    H1[HEDT Ref Build<br/>TR PRO + dual 5090 + dual p150a]
    H2[NVMe RAID Tiers + Telemetry]
    H3[Deterministic Orchestration<br/>PCIe lanes, thermal, power]
  end

  subgraph SW[Software Stack]
    S1["Agentic RAG Core<br/>Intake to Interestingness I of s to Route"]
    S2[Calisota Insight SaaS<br/>CoT verification, audits]
    S3[MCP Server-as-a-Service<br/>context and interop bus]
    S4[Doc Intelligence Tooling<br/>OCR, triage, classifiers]
  end

  HW -->|Low-latency, on-prem SLO| SW

  %% ===== GTM / Offers =====
  subgraph GO[Commercial Offers]
    O1[Appliance & Reference BOM<br/>validated builds]
    O2[Enterprise SaaS - per-seat/usage]
    O3[Private Cloud / Managed Service]
    O4[Enterprise License & Support<br/>on-prem]
    O5[Consulting / Integration<br/>hybrid routing, compliance]
    O6[Marketplace Add-ons<br/>adapters, policies, plugins]
    O7[Training & Certification]
    O8[Research & Intelligence Products<br/>Synthetic Curiosity packs]
  end

  SW --> O2
  SW --> O3
  SW --> O4
  SW --> O6
  SW --> O7
  SW --> O8

  HW --> O1
  HW --> O3
  HW --> O4
  S4 --> O8

  %% ===== Revenue Outcomes =====
  subgraph REV[Revenue Streams - best-case 10× mix]
    R1[ARR: Insight SaaS<br/>Seats + usage tiers]
    R2[MRR: Managed Service<br/>SLOs, data-locality premiums]
    R3[License & Support<br/>Annual + TSAs]
    R4[Appliance Margin<br/>Ref BOM kits & services]
    R5[Consulting & Integration<br/>Fixed + T&M]
    R6[Marketplace Rev-Share<br/>Adapters/policies]
    R7[Training/Certs<br/>Academy + exams]
    R8[Research & Intelligence<br/>Briefings, datasets]
  end

  O2 --> R1
  O3 --> R2
  O4 --> R3
  O1 --> R4
  O5 --> R5
  O6 --> R6
  O7 --> R7
  O8 --> R8

  %% ===== 10× Scaling Levers =====
  subgraph X10[10× Scaling Levers]
    L1[Land-and-Expand: Seat growth & feature gates]
    L2[SLO Tiers: Gold/Platinum latency & privacy]
    L3[Vertical Packs: Regulated templates - HIPAA/FIN/DoD]
    L4[Partner Channel: SI & OEM co-sell]
    L5[Marketplace Flywheel: Third-party adapters]
    L6[Telemetry-to-Tuning: Auto-optimization upsells]
    L7[Proof Bundles: Audited before/after hallucination cuts]
    L8[Hardware Refresh Cycles: 24–36 mo swap + services]
  end

  R1 --- L1
  R2 --- L2
  R3 --- L3
  R4 --- L8
  R5 --- L4
  R6 --- L5
  R7 --- L3
  R8 --- L7

  %% ===== KPI Anchors =====
  classDef kpi fill:#1a4d2e,stroke:#4ade80,stroke-width:2px,color:#e0ffe0
  K1((LTV/CAC > 5)):::kpi
  K2((Gross Margin: SW > 80%)):::kpi
  K3((Attach Rate: MSP > 30%)):::kpi
  K4((Renewals > 95% logo)):::kpi

  R1 --- K2
  R2 --- K3
  R3 --- K4
  R4 --- K1

  %% ===== Styling for Dark Background =====
  classDef hwStyle fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#e0f2fe
  classDef swStyle fill:#4c1d95,stroke:#a78bfa,stroke-width:2px,color:#ede9fe
  classDef goStyle fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fed7aa
  classDef revStyle fill:#134e4a,stroke:#5eead4,stroke-width:2px,color:#ccfbf1
  classDef x10Style fill:#831843,stroke:#f472b6,stroke-width:2px,color:#fce7f3

  class H1,H2,H3 hwStyle
  class S1,S2,S3,S4 swStyle
  class O1,O2,O3,O4,O5,O6,O7,O8 goStyle
  class R1,R2,R3,R4,R5,R6,R7,R8 revStyle
  class L1,L2,L3,L4,L5,L6,L7,L8 x10Style

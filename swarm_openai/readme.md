# Swarm: A Simple Multi-Agent Orchestration Library (https://github.com/openai/swarm?tab=readme-ov-file)

![Swarm Logo](assets/logo.png)

Welcome to Swarm! 🐝 This library makes it easy to create, manage, and handoff conversations between multiple agents using the power of language models like GPT-4.

## Why Swarm?

Swarm is designed to be:

1. **Simple**: Easy to set up and use, with a minimal learning curve.
2. **Flexible**: Customize agents, tools, and conversation flows to suit your needs.
3. **Controllable**: Explicitly manage agent handoffs and context variables.

## Getting Started

### Installation

Install Swarm using pip:

```
pip install git+https://github.com/openai/swarm.git
```

Or clone the repository:

```bash
git clone https://github.com/openai/swarm.git
cd swarm
pip install -e .
```

### Usage

Here's a simple example of creating two agents and running a conversation between them:

```python
from swarm import Swarm, Agent

# Define agents with instructions and tools (functions)
refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[lambda item_name: f"Refunded {item_name}!"],
)

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the products.",
    tools=[
        lambda product, price: f"Order confirmed! {product} for ${price}."
    ],
)

# Initialize Swarm client
client = Swarm()

messages = []
user_query = "I want to return this item."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Run conversation with first agent
response = client.run(agent=sales_assistant, messages=messages)
messages.extend(response.messages)

user_query = "Actually, I want a refund."  # Implicitly refers to the last item
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Run conversation with second agent
response = client.run(agent=refund_agent, messages=messages)
messages.extend(response.messages)
```

## Key Concepts

### Agents

Agents represent autonomous entities with instructions and tools (functions). They can be handed off to other agents when necessary.

### Tools (Functions)

Tools are Python functions that agents can call to perform tasks. Swarm automatically converts these functions into JSON Schema for use with the language model.

### Context Variables

Context variables allow agents to share and update state between turns in a conversation. They can be accessed and modified using special context variables.

### Handoffs

Handoffs allow agents to transfer control to another agent when they encounter a task outside their purview. This is done by returning an instance of the new agent from a tool function.

## Examples

Check out the [examples](examples/) directory for more complex use cases, including:

- airline: A multi-agent setup for handling different customer service requests in an airline context.
- support_bot: A customer service bot with a user interface agent and a help center agent with several tools.
- personal_shopper: A personal shopping assistant that can make sales and refund orders.

## Contributing

Contributions are welcome! If you encounter any issues or have ideas for improvements, please submit an issue or pull request. For more information on contributing, see the [contributing guidelines](CONTRIBUTING.md).

## License

Swarm is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

_This README was generated with 💕 by the Swarm team at ```yaml
appVersion: 0.3.3
appBuildVersion: "1"
modelPath: bartowski/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q6_K.gguf
prediction:
  layers:
    - layerName: modelDefault
      config:
        fields:
          - key: llm.prediction.promptTemplate
            value: <Default prompt template omitted for brevity>
          - key: llm.prediction.llama.cpuThreads
            value: 12
    - layerName: userModelDefault
      config:
        fields: []
    - layerName: instance
      config:
        fields: []
    - layerName: conversationSpecific
      config:
        fields:
          - key: llm.prediction.temperature
            value: 0.51
load:
  layers:
    - layerName: currentlyLoaded
      config:
        fields:
          - key: llm.load.contextLength
            value: 17596
          - key: llm.load.llama.acceleration.offloadRatio
            value: 1
    - layerName: instance
      config:
        fields: []
hardware:
  gpuSurveyResult:
    result:
      code: Success
      message: ""
    gpuInfo:
      - name: NVIDIA GeForce RTX 3080
        deviceId: 0
        totalMemoryCapacityBytes: 112228052992
        dedicatedMemoryCapacityBytes: 10995367936
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3060
        deviceId: 2
        totalMemoryCapacityBytes: 114117586944
        dedicatedMemoryCapacityBytes: 12884901888
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3060
        deviceId: 4
        totalMemoryCapacityBytes: 114117586944
        dedicatedMemoryCapacityBytes: 12884901888
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3080
        deviceId: 5
        totalMemoryCapacityBytes: 112228052992
        dedicatedMemoryCapacityBytes: 10995367936
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3060
        deviceId: 7
        totalMemoryCapacityBytes: 114117586944
        dedicatedMemoryCapacityBytes: 12884901888
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3080
        deviceId: 8
        totalMemoryCapacityBytes: 112228052992
        dedicatedMemoryCapacityBytes: 10995367936
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3060
        deviceId: 10
        totalMemoryCapacityBytes: 114117586944
        dedicatedMemoryCapacityBytes: 12884901888
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
      - name: NVIDIA GeForce RTX 3080
        deviceId: 11
        totalMemoryCapacityBytes: 112228052992
        dedicatedMemoryCapacityBytes: 10995367936
        integrationType: Discrete
        detectionPlatform: Vulkan
        detectionPlatformVersion: 1.3.283
        otherInfo: {}
  cpuSurveyResult:
    result:
      code: Success
      message: ""
    cpuInfo:
      architecture: x86_64
      supportedInstructionSetExtensions:
        - AVX2
        - AVX
selectedRuntimes:
  - modelCompatibilityType: gguf
    runtime:
      name: llama.cpp-linux-x86_64-nvidia-cuda-avx2
      version: 1.1.10
```

Here's an example of a conversation between two agents, `sales_agent` and `refund_agent`, using explicit agent handoffs in Python:

```python
from swarm import Agent, Swarm

# Define agents with instructions and tools (functions)
def transfer_to_refund_agent():
    return RefundAgent()

class SalesAgent:
    def __init__(self):
        self.name = "Sales Assistant"
        self.instructions = (
            "You are a sales assistant. Sell the products."
            "If the user wants to return an item, hand off to the refund agent."
        )
        self.tools = [
            lambda product, price: f"Order confirmed! {product} for ${price}.",
            transfer_to_refund_agent,
        ]

class RefundAgent:
    def __init__(self):
        self.name = "Refund Agent"
        self.instructions = (
            "You are a refund agent. Help the user with refunds."
            "Once done, hand back to the sales agent."
        )
        self.tools = [lambda item_name: f"Refunded {item_name}!", transfer_to_sales_agent]

# Initialize Swarm client
client = Swarm()

messages = []
user_query = "I want to return this item."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Run conversation with sales agent
response = client.run(agent=SalesAgent(), messages=messages)
messages.extend(response.messages)

if response.agent.name == "Refund Agent":
    print("Handoff to refund agent successful.")
else:
    print("Unexpected agent after handoff.")

user_query = "Thanks! Anything else I can do?"
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Run conversation with refund agent
response = client.run(agent=RefundAgent(), messages=messages)
messages.extend(response.messages)

if response.agent.name == "Sales Assistant":
    print("Handoff back to sales agent successful.")
else:
    print("Unexpected agent after handoff.")
```

In this example:

1. `sales_agent` and `refund_agent` are defined as classes with their respective instructions and tools.
2. The `transfer_to_refund_agent` and `transfer_to_sales_agent` functions are used to explicitly hand off control between agents.
3. The conversation starts with the user interacting with `sales_agent`.
4. When the user expresses a desire to return an item, `sales_agent` hands off control to `refund_agent`.
5. After handling the refund, `refund_agent` hands back control to `sales_agent`.

This example demonstrates how agent handoffs can be used to manage complex conversation flows efficiently using Swarm.

 In plain English, the given formula can be explained as follows:

**For every individual in the population (x), there must exist at least one person who is both clever (C(y)) and speaks the language (S(y)).**

In other words, no matter how diverse or large the population is, there will always be someone who possesses both of these qualities.

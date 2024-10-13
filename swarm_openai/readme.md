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

_This README was generated with 💕 by the Swarm team at OpenAI._

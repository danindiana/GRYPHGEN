# Example GRYPHGEN Project

This is an example project demonstrating how to use the GRYPHGEN grid computing framework.

## Project Structure

```
example_project/
├── README.md           # This file
├── project.yml         # Project configuration
├── main.py             # Main application
├── requirements.txt    # Python dependencies
└── tasks/              # Task definitions
    ├── preprocess.py
    ├── train.py
    └── evaluate.py
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the project**:
   Edit `project.yml` to match your requirements.

3. **Run the project**:
   ```bash
   python main.py
   ```

## Configuration

The `project.yml` file contains project-specific configuration:

- **name**: Project identifier
- **description**: Project description
- **tasks**: Task definitions and dependencies
- **resources**: Default resource requirements

## Task Definitions

Tasks are defined as Python modules in the `tasks/` directory. Each task should implement:

```python
async def run(context):
    """Execute the task.

    Args:
        context: Task execution context with resources and configuration

    Returns:
        Task result dictionary
    """
    # Your task implementation
    pass
```

## Example Workflow

This example implements a simple ML pipeline:

1. **Preprocess**: Load and preprocess data (CPU-intensive)
2. **Train**: Train model on GPU (GPU-intensive)
3. **Evaluate**: Evaluate model performance (GPU-intensive)

## Resource Requirements

- **Preprocess**: 4 CPU cores, 8GB RAM
- **Train**: 1 GPU (12GB VRAM), 2 CPU cores, 16GB RAM
- **Evaluate**: 1 GPU (4GB VRAM), 1 CPU core, 4GB RAM

## Running with GRYPHGEN

```python
import asyncio
from GRYPHGEN import GryphgenFramework
from SYMORG import TaskPriority

async def main():
    framework = GryphgenFramework("config.yml")
    await framework.initialize()

    # Submit tasks from this project
    await framework.scheduler.submit_task(
        "preprocess",
        "Data Preprocessing",
        resources_required={"cpu": 4.0, "memory": 8 * 1024**3},
        priority=TaskPriority.HIGH
    )

    await framework.scheduler.submit_task(
        "train",
        "Model Training",
        resources_required={"gpu_0": 12 * 1024**3, "cpu": 2.0},
        dependencies=["preprocess"],
        priority=TaskPriority.CRITICAL
    )

    await framework.scheduler.submit_task(
        "evaluate",
        "Model Evaluation",
        resources_required={"gpu_0": 4 * 1024**3, "cpu": 1.0},
        dependencies=["train"],
        priority=TaskPriority.HIGH
    )

    await framework.start()

asyncio.run(main())
```

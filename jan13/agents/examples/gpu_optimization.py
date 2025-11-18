"""GPU optimization example for NVIDIA RTX 4080."""

import asyncio
import logging
from src.agents.infrastructure import InfrastructureAgent
from src.models.config import DeploymentConfig
from src.utils.monitoring import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def print_gpu_info():
    """Print GPU information."""
    gpu_monitor = GPUMonitor(gpu_id=0)
    stats = gpu_monitor.get_stats()

    if not stats:
        logger.warning("GPU not available or nvidia-smi not installed")
        return

    logger.info("=" * 60)
    logger.info("GPU Information (NVIDIA RTX 4080)")
    logger.info("=" * 60)
    logger.info(f"GPU ID: {stats['gpu_id']}")
    logger.info(f"Memory Total: {stats['memory_total']} MB")
    logger.info(f"Memory Free: {stats['memory_free']} MB")
    logger.info(f"Memory Used: {stats['memory_used']} MB")
    logger.info(f"Temperature: {stats['temperature']}°C")
    logger.info(f"Power Usage: {stats['power_usage']:.2f}W")
    logger.info(f"Graphics Clock: {stats['graphics_clock']} MHz")
    logger.info(f"Memory Clock: {stats['memory_clock']} MHz")
    logger.info("=" * 60)


async def benchmark_gpu_performance(agent: InfrastructureAgent, duration: int = 60):
    """Benchmark GPU performance during LLM inference.

    Args:
        agent: Infrastructure agent
        duration: Benchmark duration in seconds
    """
    logger.info(f"\nRunning GPU benchmark for {duration}s...")

    gpu_monitor = GPUMonitor(gpu_id=0)

    max_utilization = 0.0
    max_memory = 0
    max_temp = 0.0
    samples = 0

    import time

    start_time = time.time()

    while time.time() - start_time < duration:
        stats = gpu_monitor.get_stats()

        if stats:
            max_utilization = max(max_utilization, stats["utilization"])
            max_memory = max(max_memory, stats["memory_used"])
            max_temp = max(max_temp, stats["temperature"])
            samples += 1

            logger.info(
                f"GPU: {stats['utilization']:.1f}% | "
                f"Memory: {stats['memory_used']}MB | "
                f"Temp: {stats['temperature']}°C"
            )

        await asyncio.sleep(2)

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Max GPU Utilization: {max_utilization:.1f}%")
    logger.info(f"Max Memory Used: {max_memory} MB")
    logger.info(f"Max Temperature: {max_temp}°C")
    logger.info(f"Samples: {samples}")
    logger.info("=" * 60)


async def main():
    """GPU optimization example."""
    logger.info("=" * 60)
    logger.info("GPU Optimization Example (NVIDIA RTX 4080)")
    logger.info("=" * 60)

    # Print GPU info
    print_gpu_info()

    # Create optimized configuration for RTX 4080
    config = DeploymentConfig()

    # Optimize for RTX 4080 (16GB VRAM)
    config.gpu.enabled = True
    config.gpu.gpu_id = 0
    config.gpu.memory_fraction = 0.9  # Use 90% of 16GB = ~14.4GB
    config.gpu.num_threads = 8  # Optimize for tensor cores
    config.gpu.vram_gb = 16

    # Ollama configuration for maximum performance
    config.ollama.num_gpu = 1
    config.ollama.gpu_memory_fraction = 0.9
    config.ollama.context_length = 4096  # Optimal for 16GB
    config.ollama.batch_size = 32  # Good balance for RTX 4080
    config.ollama.models = ["llama2"]  # Start with one model

    logger.info("\nGPU Configuration:")
    logger.info(f"  Memory Fraction: {config.gpu.memory_fraction * 100:.0f}%")
    logger.info(f"  Available VRAM: ~{config.gpu.vram_gb * config.gpu.memory_fraction:.1f}GB")
    logger.info(f"  Threads: {config.gpu.num_threads}")
    logger.info(f"  Context Length: {config.ollama.context_length}")
    logger.info(f"  Batch Size: {config.ollama.batch_size}")

    # Create infrastructure agent
    agent = InfrastructureAgent(config)

    try:
        # Deploy
        logger.info("\nDeploying with GPU optimizations...")
        success = await agent.deploy()

        if not success:
            logger.error("Deployment failed")
            return

        logger.info("Deployment successful!")

        # Run benchmark
        await benchmark_gpu_performance(agent, duration=60)

        # Get final status
        status = await agent.get_status()

        if status.metrics and status.metrics.gpu_utilization is not None:
            logger.info("\nFinal GPU Status:")
            logger.info(f"  Utilization: {status.metrics.gpu_utilization:.1f}%")
            logger.info(f"  Memory Used: {status.metrics.gpu_memory_used} MB")
            logger.info(f"  Temperature: {status.metrics.gpu_temperature}°C")

        logger.info("\nOptimization Tips for RTX 4080:")
        logger.info("  1. Use 90% memory fraction (leave headroom for system)")
        logger.info("  2. Enable tensor cores with optimal thread count")
        logger.info("  3. Use 4-bit quantization (q4_k_m) for larger models")
        logger.info("  4. Context length of 4096 balances performance and capability")
        logger.info("  5. Batch size 32 optimal for 16GB VRAM")
        logger.info("  6. Monitor temperature - keep under 80°C for longevity")

    except KeyboardInterrupt:
        logger.info("\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await agent.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

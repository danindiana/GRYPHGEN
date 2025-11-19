"""
SYMORQ Orchestration Module

LLM-based orchestration using ZeroMQ message passing for distributed
grid computing coordination.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import zmq
import zmq.asyncio
from loguru import logger


class Orchestrator:
    """Main orchestration component using LLM and ZeroMQ."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.context = zmq.asyncio.Context()
        self.is_running = False

        # ZeroMQ sockets
        self.command_socket: Optional[zmq.asyncio.Socket] = None
        self.pub_socket: Optional[zmq.asyncio.Socket] = None
        self.sub_socket: Optional[zmq.asyncio.Socket] = None

        # State management
        self.tasks: Dict[str, Dict] = {}
        self.resources: Dict[str, Dict] = {}
        self.workers: Dict[str, Dict] = {}

    async def initialize(self):
        """Initialize orchestration system."""
        logger.info("Initializing Orchestrator...")

        # Setup ZeroMQ sockets
        zmq_config = self.config.get("zeromq", {})

        # Command socket (ROUTER) for receiving commands
        self.command_socket = self.context.socket(zmq.ROUTER)
        self.command_socket.bind(f"tcp://*:{zmq_config.get('orchestrator_port', 5555)}")

        # Publisher socket for broadcasting events
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{zmq_config.get('pub_port', 5558)}")

        # Subscriber socket for receiving updates
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://localhost:{zmq_config.get('sub_port', 5559)}")
        self.sub_socket.subscribe(b"")

        logger.info("✓ Orchestrator initialized")

    async def start(self):
        """Start the orchestrator."""
        self.is_running = True
        logger.info("Starting Orchestrator...")

        # Start background tasks
        await asyncio.gather(
            self._handle_commands(),
            self._handle_updates(),
            self._monitor_resources(),
            self._process_tasks(),
        )

    async def shutdown(self):
        """Shutdown the orchestrator."""
        logger.info("Shutting down Orchestrator...")
        self.is_running = False

        # Close sockets
        if self.command_socket:
            self.command_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.sub_socket:
            self.sub_socket.close()

        # Terminate context
        self.context.term()

        logger.info("✓ Orchestrator shutdown complete")

    async def _handle_commands(self):
        """Handle incoming commands."""
        while self.is_running:
            try:
                # Receive command with timeout
                if await self.command_socket.poll(timeout=1000):
                    identity, _, message = await self.command_socket.recv_multipart()

                    try:
                        command = json.loads(message.decode())
                        logger.debug(f"Received command: {command.get('type')}")

                        response = await self._process_command(command)

                        # Send response
                        await self.command_socket.send_multipart([
                            identity,
                            b"",
                            json.dumps(response).encode()
                        ])
                    except Exception as e:
                        logger.error(f"Error processing command: {e}")
                        await self.command_socket.send_multipart([
                            identity,
                            b"",
                            json.dumps({"error": str(e)}).encode()
                        ])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in command handler: {e}")
                await asyncio.sleep(1)

    async def _process_command(self, command: Dict) -> Dict:
        """Process a command and return response.

        Args:
            command: Command dictionary

        Returns:
            Response dictionary
        """
        cmd_type = command.get("type")

        if cmd_type == "submit_task":
            return await self.submit_task(command.get("task", {}))
        elif cmd_type == "cancel_task":
            return await self.cancel_task(command.get("task_id"))
        elif cmd_type == "register_worker":
            return await self.register_worker(command.get("worker_info", {}))
        elif cmd_type == "get_status":
            return self.get_status()
        else:
            return {"error": f"Unknown command type: {cmd_type}"}

    async def submit_task(self, task: Dict) -> Dict:
        """Submit a new task for execution.

        Args:
            task: Task specification

        Returns:
            Task submission response
        """
        task_id = task.get("id", f"task_{datetime.now().timestamp()}")

        self.tasks[task_id] = {
            "id": task_id,
            "spec": task,
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        logger.info(f"Task {task_id} submitted")

        # Broadcast task submission
        await self.pub_socket.send_multipart([
            b"task.submitted",
            json.dumps({"task_id": task_id, "task": task}).encode()
        ])

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending"
        }

    async def cancel_task(self, task_id: str) -> Dict:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            Cancellation response
        """
        if task_id not in self.tasks:
            return {"success": False, "error": f"Task {task_id} not found"}

        self.tasks[task_id]["status"] = "cancelled"
        self.tasks[task_id]["updated_at"] = datetime.now().isoformat()

        logger.info(f"Task {task_id} cancelled")

        # Broadcast cancellation
        await self.pub_socket.send_multipart([
            b"task.cancelled",
            json.dumps({"task_id": task_id}).encode()
        ])

        return {"success": True, "task_id": task_id}

    async def register_worker(self, worker_info: Dict) -> Dict:
        """Register a new worker.

        Args:
            worker_info: Worker information

        Returns:
            Registration response
        """
        worker_id = worker_info.get("id", f"worker_{len(self.workers)}")

        self.workers[worker_id] = {
            "id": worker_id,
            "info": worker_info,
            "status": "active",
            "registered_at": datetime.now().isoformat(),
        }

        logger.info(f"Worker {worker_id} registered")

        return {
            "success": True,
            "worker_id": worker_id
        }

    async def _handle_updates(self):
        """Handle incoming updates from workers and schedulers."""
        while self.is_running:
            try:
                if await self.sub_socket.poll(timeout=1000):
                    topic, message = await self.sub_socket.recv_multipart()

                    try:
                        update = json.loads(message.decode())
                        logger.debug(f"Received update on {topic.decode()}: {update}")

                        await self._process_update(topic.decode(), update)
                    except Exception as e:
                        logger.error(f"Error processing update: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update handler: {e}")
                await asyncio.sleep(1)

    async def _process_update(self, topic: str, update: Dict):
        """Process an update message.

        Args:
            topic: Update topic
            update: Update data
        """
        if topic.startswith("task."):
            task_id = update.get("task_id")
            if task_id and task_id in self.tasks:
                self.tasks[task_id].update(update)
                self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
        elif topic.startswith("worker."):
            worker_id = update.get("worker_id")
            if worker_id and worker_id in self.workers:
                self.workers[worker_id].update(update)
        elif topic.startswith("resource."):
            resource_id = update.get("resource_id")
            if resource_id:
                self.resources[resource_id] = update

    async def _monitor_resources(self):
        """Monitor system resources."""
        while self.is_running:
            try:
                # Collect resource information
                resource_status = {
                    "timestamp": datetime.now().isoformat(),
                    "tasks": len(self.tasks),
                    "workers": len(self.workers),
                    "resources": len(self.resources),
                }

                # Broadcast resource status
                await self.pub_socket.send_multipart([
                    b"resources.status",
                    json.dumps(resource_status).encode()
                ])

                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(30)

    async def _process_tasks(self):
        """Process pending tasks."""
        while self.is_running:
            try:
                # Find pending tasks
                pending_tasks = [
                    task_id for task_id, task in self.tasks.items()
                    if task["status"] == "pending"
                ]

                for task_id in pending_tasks:
                    # Here you would use LLM to determine best resource allocation
                    # For now, just mark as processing
                    self.tasks[task_id]["status"] = "processing"
                    self.tasks[task_id]["updated_at"] = datetime.now().isoformat()

                    logger.info(f"Processing task {task_id}")

                await asyncio.sleep(5)  # Process every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
                await asyncio.sleep(5)

    def get_status(self) -> Dict:
        """Get current orchestrator status.

        Returns:
            Status dictionary
        """
        return {
            "running": self.is_running,
            "tasks": {
                "total": len(self.tasks),
                "pending": sum(1 for t in self.tasks.values() if t["status"] == "pending"),
                "processing": sum(1 for t in self.tasks.values() if t["status"] == "processing"),
                "completed": sum(1 for t in self.tasks.values() if t["status"] == "completed"),
                "failed": sum(1 for t in self.tasks.values() if t["status"] == "failed"),
            },
            "workers": len(self.workers),
            "resources": len(self.resources),
        }


def main():
    """Main entry point for orchestration module."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.settings import load_config

    async def run():
        config = load_config()
        orchestrator = Orchestrator(config)

        try:
            await orchestrator.initialize()
            await orchestrator.start()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await orchestrator.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()

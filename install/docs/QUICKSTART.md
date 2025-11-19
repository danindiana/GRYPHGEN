# GRYPHGEN Quick Start Guide

## Prerequisites

- Ubuntu 22.04 LTS or 24.04 LTS
- NVIDIA RTX 4080 16GB (or compatible GPU)
- 32GB+ RAM (recommended)
- 500GB+ storage
- Sudo/root access

## Installation Methods

### Method 1: Automated Installation (Recommended)

Complete installation with a single command sequence:

```bash
# Navigate to install directory
cd install

# 1. Install base system
sudo bash scripts/install_base.sh

# 2. Install NVIDIA CUDA (may require reboot)
sudo bash scripts/install_cuda.sh

# 3. Install Docker with NVIDIA support
sudo bash scripts/install_docker.sh

# 4. Install Python environment
bash scripts/install_python.sh

# 5. Install LLM components
bash scripts/install_llm.sh

# 6. Validate installation
bash scripts/validate_environment.sh
```

### Method 2: Docker Deployment

For containerized deployment:

```bash
# Build images
cd docker
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f gryphgen-llm
```

## Post-Installation

### Activate Python Environment

```bash
# Activate GRYPHGEN environment
gryphgen-activate

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Test GPU Access

```bash
# Test NVIDIA driver
nvidia-smi

# Test Docker GPU access
test-docker-gpu

# Run comprehensive tests
cd tests
python test_cuda.py
python test_environment.py
python benchmark.py
```

### Download Models

```bash
# Activate environment
gryphgen-activate

# Login to Hugging Face (optional)
huggingface-cli login

# Download a model (example: Mistral 7B)
python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Save to local directory
tokenizer.save_pretrained("/models/gryphgen/mistral/7B")
model.save_pretrained("/models/gryphgen/mistral/7B")
EOF
```

## Common Operations

### Start Jupyter Lab

```bash
gryphgen-activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### Run FastAPI Server

```bash
gryphgen-activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Monitor GPU

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check temperatures
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# Monitor power usage
nvidia-smi --query-gpu=power.draw --format=csv,noheader
```

## Directory Structure

```
/opt/gryphgen/          # GRYPHGEN home
├── venv/              # Python virtual environment
├── configs/           # Configuration files
└── pytorch_config.py  # PyTorch configuration

/models/gryphgen/      # Model storage
├── llama/            # LLaMA models
├── mistral/          # Mistral models
├── codellama/        # CodeLLaMA models
└── embeddings/       # Embedding models

/data/gryphgen/        # Data storage
├── vector_db/        # Vector database
├── cache/            # Cache files
└── logs/             # Log files
```

## Environment Variables

Add to your `~/.bashrc`:

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# GRYPHGEN
export GRYPHGEN_HOME=/opt/gryphgen
export GRYPHGEN_DATA=/data/gryphgen
export GRYPHGEN_MODELS=/models/gryphgen

# Hugging Face
export HF_HOME=/models/gryphgen/cache
export TRANSFORMERS_CACHE=/models/gryphgen/cache
```

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
nvcc --version

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker run hello-world
```

### PyTorch Can't Find CUDA

```bash
# Reinstall PyTorch with correct CUDA version
gryphgen-activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Out of Memory

```bash
# Set memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or in Python:
import torch
torch.cuda.empty_cache()
```

## Performance Optimization

### GPU Settings

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Set power limit (320W for RTX 4080)
sudo nvidia-smi -pl 320

# Set compute mode
sudo nvidia-smi -c 0
```

### PyTorch Settings

```python
import torch

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True
```

## Next Steps

1. **Read the documentation**: Check `docs/` for detailed guides
2. **Run examples**: Explore example notebooks and scripts
3. **Join the community**: Contribute to the project
4. **Optimize for your use case**: Tune settings for your specific workload

## Support

- **Documentation**: [Full docs](../README.md)
- **Issues**: [GitHub Issues](https://github.com/danindiana/GRYPHGEN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danindiana/GRYPHGEN/discussions)

## Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Docker Documentation](https://docs.docker.com/)

---

**Happy building with GRYPHGEN!** 🚀

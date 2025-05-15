import modal

# Build a Docker image with Python 3.12, vLLM, Hugging Face Hub, and flashinfer-python
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# Enable vLLM V1 engine for better performance
vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

# Model configuration
MODEL_NAME = "ibm-granite/granite-3.2-8b-instruct-preview"
# No revision specified since we're using the latest version

# Volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create Modal app
app = modal.App("granite-vllm-openai-compatible")

N_GPU = 1  # Use 1 GPU
API_KEY = "super-secret-key"  # For production, use modal.Secret
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    allow_concurrent_inputs=100,
    scaledown_window=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    # Added --max-model-len=16384 to set KV cache to 16K tokens
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
        "--max-model-len=16384",
        # "--kv_cache_dtype", "fp8",
        "--gpu-memory-utilization",
        "0.9",
        # "--enable_chunked_prefill", "True",
    ]

    subprocess.Popen(" ".join(cmd), shell=True)

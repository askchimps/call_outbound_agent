# Deployment Guide

This guide provides step-by-step instructions to deploy:
1. **vLLM Server** on Vast.ai (for LLM inference)
2. **LiveKit Agent** on AWS EC2 (for voice calling)

---

## Prerequisites

Before starting, ensure you have:
- [ ] Vast.ai account with credits
- [ ] AWS account with EC2 access
- [ ] LiveKit Cloud account (https://cloud.livekit.io)
- [ ] Deepgram API key (https://console.deepgram.com)
- [ ] Cartesia API key (https://play.cartesia.ai) OR ElevenLabs API key
- [ ] SIP trunk configured in LiveKit Cloud

---

## Part 1: Deploy vLLM on Vast.ai

### Step 1.1: Create Vast.ai Account and Add Credits
```bash
# Go to https://vast.ai and sign up
# Add credits via Settings → Billing (minimum $10 recommended)
```

### Step 1.2: Find and Rent a GPU Instance

1. Go to https://vast.ai/console/create/
2. Click **Templates** tab
3. Search for **vLLM** in the template search
4. Select the vLLM template
5. Filter by GPU:
   - **RTX 3090 (24GB)** - Good for Qwen 7B (~$0.20/hr)
   - **RTX 4090 (24GB)** - Faster (~$0.40/hr)
   - **RTX 5090 (32GB)** - Best performance (~$0.80/hr)
6. Filter by region (for lowest latency to your EC2):
   - Singapore/Hong Kong for Asia
   - US East for Americas
7. Click **RENT** on your chosen instance
8. Wait 2-5 minutes for instance to start

### Step 1.3: Get SSH Connection Details

From Vast.ai dashboard, find your instance and note:
- **SSH Host**: e.g., `ssh8.vast.ai`
- **SSH Port**: e.g., `12345`
- **Direct Port**: e.g., `8000` → mapped to `54321`

The connection command is shown in the dashboard.

### Step 1.4: Connect to Your Instance
```bash
# Copy the SSH command from Vast.ai dashboard
ssh -p <PORT> root@<SSH_HOST>

# Example:
ssh -p 12345 root@ssh8.vast.ai
```

### Step 1.5: Stop Default Model (if running)
```bash
# Check if vLLM is already running
ps aux | grep vllm

# Kill any existing vLLM process
pkill -f vllm

# Verify it's stopped
ps aux | grep vllm
```

### Step 1.6: Install tmux (for persistent sessions)
```bash
apt update && apt install -y tmux
```

### Step 1.7: Start vLLM with Optimized Settings
```bash
# Create a new tmux session
tmux new -s vllm

# Start vLLM with optimized flags
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-log-requests

# Wait for "Uvicorn running on http://0.0.0.0:8000" message
# Then detach: Press Ctrl+B, then D
```

**vLLM Flags Explained:**
| Flag | Purpose |
|------|---------|
| `--quantization awq` | Use 4-bit quantization (faster, less memory) |
| `--enable-prefix-caching` | Cache system prompts (~80% cache hit rate) |
| `--max-model-len 4096` | Limit context length (saves memory) |
| `--gpu-memory-utilization 0.90` | Use 90% of GPU memory |
| `--enable-auto-tool-choice` | Enable automatic tool/function calling |
| `--tool-call-parser hermes` | Use Hermes format for tool calls (works with Qwen) |
| `--disable-log-requests` | Reduce log noise |

### Step 1.8: Verify vLLM is Running
```bash
# Test locally
curl http://localhost:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"Qwen/Qwen2.5-7B-Instruct-AWQ",...}]}
```

### Step 1.9: Test from Outside (Important!)
```bash
# From your local machine, test the public endpoint
# Get the public URL from Vast.ai dashboard (Open Ports section)

curl http://<VAST_PUBLIC_IP>:<MAPPED_PORT>/v1/models

# Example:
curl http://209.20.158.140:54321/v1/models
```

### Step 1.10: Note Your LLM Endpoint

Save this URL - you'll need it for EC2 configuration:
```
LLM_BASE_URL=http://<VAST_PUBLIC_IP>:<MAPPED_PORT>/v1

# Example:
LLM_BASE_URL=http://209.20.158.140:54321/v1
```

### tmux Quick Reference
```bash
# List all sessions
tmux ls

# Attach to vllm session (view logs)
tmux attach -t vllm

# Detach (keep running): Ctrl+B, then D

# Kill session (stop vLLM)
tmux kill-session -t vllm

# Scroll logs: Ctrl+B, then [ (arrow keys to scroll, q to exit)
```

---

## Part 2: Deploy LiveKit Agent on AWS EC2

### Step 2.1: Launch EC2 Instance

1. Go to AWS Console → EC2 → **Launch Instance**
2. Configure:
   - **Name**: `livekit-agent`
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance type**: `t3.medium` (2 vCPU, 4GB RAM) or larger
   - **Key pair**: Create new or select existing
   - **Network settings**:
     - Allow SSH from your IP
     - Allow HTTP (port 8000) from anywhere
3. Click **Launch Instance**
4. Wait for instance to be "Running"

### Step 2.2: Configure Security Group

Go to EC2 → Security Groups → Select your instance's security group → Edit inbound rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | My IP | SSH access |
| Custom TCP | 8000 | 0.0.0.0/0 | API endpoint |

### Step 2.3: Connect to EC2
```bash
# Download your key pair (.pem file) if you haven't

# Set correct permissions
chmod 400 your-key.pem

# Connect to EC2
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Example:
ssh -i livekit-agent.pem ubuntu@54.169.123.45
```

### Step 2.4: Install Docker and Docker Compose
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io

# Install Docker Compose
sudo apt install -y docker-compose

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu

# Apply group changes (or logout and login again)
newgrp docker

# Verify installation
docker --version
docker-compose --version
```

### Step 2.5: Clone Repository
```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/outbound-caller-python.git

# Navigate to project directory
cd outbound-caller-python

# List files to verify
ls -la
```

### Step 2.6: Create Environment File
```bash
# Copy example file
cp .env.example .env.local

# Edit the file
nano .env.local
```

### Step 2.7: Configure Environment Variables

Fill in ALL values in `.env.local`:

```bash
# === LiveKit Configuration ===
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# === Speech-to-Text (Deepgram) ===
DEEPGRAM_API_KEY=your_deepgram_api_key

# === LLM Configuration ===
# For OpenAI:
OPENAI_API_KEY=your_openai_key
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=

# For self-hosted vLLM (from Part 1):
OPENAI_API_KEY=not-needed
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_BASE_URL=http://<VAST_PUBLIC_IP>:<PORT>/v1

# === SIP Configuration ===
SIP_OUTBOUND_TRUNK_ID=your_sip_trunk_id

# === TTS Configuration ===
# Choose provider: "cartesia" (faster) or "elevenlabs" (better quality)
TTS_PROVIDER=cartesia

# ElevenLabs settings (if TTS_PROVIDER=elevenlabs)
ELEVEN_API_KEY=your_elevenlabs_key
TTS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
TTS_MODEL_ID=eleven_flash_v2_5

# Cartesia settings (if TTS_PROVIDER=cartesia)
CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=a0e99841-438c-4a64-b679-ae501e7d6091
CARTESIA_MODEL=sonic-2
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

### Step 2.8: Create Agent Prompt
```bash
# Create/edit the base prompt file
nano baseprompt.txt
```

Paste your agent's system prompt, then save and exit.

### Step 2.9: Build and Start the Agent
```bash
# Build and run in detached mode
docker-compose up -d --build

# This will:
# 1. Build the Docker image
# 2. Start the container
# 3. Run in background
```

### Step 2.10: Verify Agent is Running
```bash
# Check container status
docker-compose ps

# Expected output:
# NAME                COMMAND             STATUS              PORTS
# outbound-caller     "python agent.py"   Up                  0.0.0.0:8000->8000/tcp

# Check logs
docker-compose logs -f

# Look for:
# "registered worker"
# "agent_name": "outbound-caller"
# "region": "Singapore South East" (or your region)

# Press Ctrl+C to exit logs
```

### Step 2.11: Test Health Endpoint
```bash
# From EC2 (local test)
curl http://localhost:8000/health

# From your local machine (remote test)
curl http://<EC2_PUBLIC_IP>:8000/health

# Expected response:
# {"status": "healthy"}
```

### Step 2.12: Setup Nginx Reverse Proxy (Optional)

Set up nginx to access the LiveKit agent from outside via EC2 public IP.

#### Install Nginx
```bash
sudo apt update
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

#### Configure Nginx
```bash
# Create nginx config
sudo nano /etc/nginx/sites-available/livekit-agent
```

Paste the following configuration:
```nginx
server {
    listen 80;
    server_name _;  # Accept any hostname/IP

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

#### Enable the Site
```bash
# Create symlink to enable site
sudo ln -s /etc/nginx/sites-available/livekit-agent /etc/nginx/sites-enabled/

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

#### Update EC2 Security Group
Make sure your EC2 security group allows:
- **Port 80** (HTTP) - for nginx access
- **Port 22** (SSH) - for management

#### Test Access
```bash
# From your local machine
curl http://<EC2_PUBLIC_IP>/health

# Expected response:
# {"status": "healthy"}
```

#### Nginx Commands Reference
```bash
# Check status
sudo systemctl status nginx

# Reload config
sudo systemctl reload nginx

# View logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### TTS Provider Comparison

| Provider | Latency | Quality | Best For |
|----------|---------|---------|----------|
| **Cartesia** | ~100ms | Good | Low latency, Asia regions |
| **ElevenLabs** | ~300-600ms | Excellent | Best voice quality |

**Recommendation:** Use Cartesia for lower latency, especially when deploying in Asia.

---

## Part 3: Test the System

### Step 3.1: Test vLLM Endpoint
```bash
# From your local machine, test the LLM
curl http://<VAST_PUBLIC_IP>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'

# Expected: JSON response with "choices" containing the model's reply
```

### Step 3.2: Test Agent Health
```bash
# From your local machine
curl http://<EC2_PUBLIC_IP>:8000/health

# Expected response:
{"status": "healthy"}
```

### Step 3.3: Make a Test Call (via API)
```bash
curl -X POST http://<EC2_PUBLIC_IP>:8000/dispatch \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+919876543210"}'

# Expected response:
{
  "dispatch_id": "AD_xxxxx",
  "room_name": "call-xxxxx",
  "phone_number": "+919876543210",
  "status": "dispatched"
}
```

### Step 3.4: Make a Test Call (via LiveKit CLI)
```bash
# Install lk CLI if not already installed
# https://docs.livekit.io/home/cli/lk/

# Configure CLI
export LIVEKIT_URL=wss://your-project.livekit.cloud
export LIVEKIT_API_KEY=your_api_key
export LIVEKIT_API_SECRET=your_api_secret

# Dispatch a call
lk dispatch create \
  --new-room \
  --agent-name outbound-caller \
  --metadata '{"phone_number": "+919876543210"}'
```

### Step 3.5: Monitor Call in Real-Time
```bash
# On EC2, watch logs
docker-compose logs -f

# Look for:
# - "registered worker" - Agent connected
# - "participant joined" - SIP call connected
# - "[PERF_LOG]" - Performance metrics
```

### Step 3.6: Check Active Rooms
```bash
# List all active rooms
lk room list

# List participants in a room
lk room participants list <room-name>
```

---

## Part 4: Common Operations

### Docker Commands (on EC2)
```bash
# View running containers
docker-compose ps

# View logs (follow mode)
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail 100

# Restart agent
docker-compose restart

# Stop agent
docker-compose down

# Rebuild and restart (after code changes)
docker-compose down && docker-compose up -d --build

# Full rebuild (clear cache)
docker-compose down --rmi all --volumes
docker system prune -af --volumes
docker-compose up -d --build
```

### vLLM Commands (on Vast.ai)
```bash
# Attach to vLLM session
tmux attach -t vllm

# Detach (keep running): Ctrl+B, then D

# View vLLM processes
ps aux | grep vllm

# Check GPU usage
nvidia-smi

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Restart vLLM
tmux kill-session -t vllm
tmux new -s vllm
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --disable-log-requests
```

### LiveKit CLI Commands
```bash
# List rooms
lk room list

# Delete a room
lk room delete <room-name>

# List SIP trunks
lk sip outbound list

# Check agent status (for Cloud Agents)
lk agent list
lk agent status <agent-id>

# Tail agent logs (for Cloud Agents)
lk agent logs <agent-id>
```

---

## Part 5: Troubleshooting

### Problem: Agent not registering with LiveKit

**Symptoms:** No "registered worker" message in logs

**Solutions:**
```bash
# 1. Check environment variables
docker-compose exec outbound-caller env | grep LIVEKIT

# 2. Verify LiveKit credentials
curl -X POST https://your-project.livekit.cloud/twirp/livekit.RoomService/ListRooms \
  -H "Authorization: Bearer $(lk token create --api-key $LIVEKIT_API_KEY --api-secret $LIVEKIT_API_SECRET)"

# 3. Check container logs
docker-compose logs --tail 50

# 4. Restart container
docker-compose restart
```

### Problem: LLM not responding

**Symptoms:** Calls connect but agent doesn't speak

**Solutions:**
```bash
# 1. Test LLM endpoint directly
curl http://<VAST_IP>:<PORT>/v1/models

# 2. Check LLM_BASE_URL in .env.local
cat .env.local | grep LLM

# 3. Test LLM completion
curl http://<VAST_IP>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "messages": [{"role": "user", "content": "Hi"}]}'

# 4. On Vast.ai, check vLLM is running
tmux attach -t vllm
# Look for errors, then Ctrl+B, D to detach
```

### Problem: SIP calls not connecting

**Symptoms:** Dispatch created but phone doesn't ring

**Solutions:**
```bash
# 1. Verify SIP trunk ID
lk sip outbound list

# 2. Check SIP_OUTBOUND_TRUNK_ID in .env.local
cat .env.local | grep SIP

# 3. Check if agent joined the room
lk room list
lk room participants list <room-name>

# 4. Check agent logs for SIP errors
docker-compose logs | grep -i sip
```

### Problem: High latency / slow responses

**Symptoms:** Long pauses between user speech and agent response

**Solutions:**
```bash
# 1. Check performance logs
docker-compose logs | grep PERF_LOG

# 2. Test LLM latency
time curl http://<VAST_IP>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "messages": [{"role": "user", "content": "Hi"}]}'

# 3. Switch to Cartesia TTS (faster than ElevenLabs)
# Edit .env.local: TTS_PROVIDER=cartesia

# 4. Check network latency to vLLM
ping <VAST_IP>
```

### Problem: vLLM out of memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```bash
# 1. Reduce max context length
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# 2. Use smaller model
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### Problem: TTS errors

**Symptoms:** Agent processes but no audio output

**Solutions:**
```bash
# 1. Check TTS configuration
cat .env.local | grep -E "TTS|CARTESIA|ELEVEN"

# 2. Verify API keys are set
# For Cartesia: CARTESIA_API_KEY
# For ElevenLabs: ELEVEN_API_KEY

# 3. Test API key validity
# Cartesia:
curl https://api.cartesia.ai/voices \
  -H "X-API-Key: your_cartesia_key"

# ElevenLabs:
curl https://api.elevenlabs.io/v1/voices \
  -H "xi-api-key: your_elevenlabs_key"
```

---

## Part 6: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHONE NETWORK                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ SIP
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LIVEKIT CLOUD (Singapore)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  SIP Trunk   │  │    Room      │  │   Agent      │               │
│  │              │──│   Server     │──│   Dispatch   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS EC2 (Agent)                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Docker Container                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │   │
│  │  │   VAD    │─▶│   STT    │─▶│   LLM    │─▶│   TTS    │      │   │
│  │  │ (Silero) │  │(Deepgram)│  │          │  │(Cartesia)│      │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP/OpenAI API
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VAST.AI (vLLM Server)                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         vLLM                                  │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │              Qwen/Qwen2.5-7B-Instruct-AWQ              │  │   │
│  │  │                    (4-bit quantized)                    │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Fine-Tuning Guide (For Later)

This section covers how to fine-tune the base model on your own data and deploy it.

### Overview

```
Base Model (FP16) → Fine-Tune with LoRA → Merge → Quantize to AWQ → Deploy
```

### Step 1: Prepare Training Data

Create a JSONL file with conversations:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant..."}, {"role": "user", "content": "Hi, I received a call?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant..."}, {"role": "user", "content": "I want to know more"}, {"role": "assistant", "content": "Of course! Let me explain..."}]}
```

**Recommended:** 1,000 - 10,000 examples for good results.

### Step 2: Setup Training Environment

Rent a GPU instance (3090/A100) or use your local machine:

```bash
pip install transformers datasets peft accelerate bitsandbytes trl
```

### Step 3: Fine-Tune with QLoRA

Create `finetune.py`:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Load base model with 4-bit quantization (for training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Training config
training_args = SFTConfig(
    output_dir="./qwen-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./qwen-finetuned-lora")
```

Run training:
```bash
python finetune.py
```

**Training time:** ~2-4 hours for 5,000 examples on a 3090.

### Step 4: Merge LoRA with Base Model

Create `merge.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (full precision)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="float16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load and merge LoRA
model = PeftModel.from_pretrained(base_model, "./qwen-finetuned-lora")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./qwen-finetuned-merged")
tokenizer.save_pretrained("./qwen-finetuned-merged")
```

Run:
```bash
python merge.py
```

### Step 5: Quantize to AWQ

Install AutoAWQ:
```bash
pip install autoawq
```

Create `quantize.py`:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./qwen-finetuned-merged"
quant_path = "./qwen-finetuned-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",  # or your own calibration dataset
)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

Run:
```bash
python quantize.py
```

**Quantization time:** ~30-60 minutes on a 3090.

### Step 6: Upload to Hugging Face (Optional)

```bash
pip install huggingface_hub
huggingface-cli login

# Upload
huggingface-cli upload your-username/qwen-finetuned-awq ./qwen-finetuned-awq
```

### Step 7: Deploy Fine-Tuned Model

On your Vast.ai instance:

```bash
# If uploaded to HuggingFace
tmux new -s vllm
vllm serve your-username/qwen-finetuned-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192

# Or if using local path
vllm serve /path/to/qwen-finetuned-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Update your `.env.local` on EC2:
```
LLM_MODEL=your-username/qwen-finetuned-awq
```

### Fine-Tuning Summary

| Step | Time | GPU Memory |
|------|------|------------|
| Fine-tune (QLoRA) | 2-4 hours | ~20GB |
| Merge LoRA | 10-20 min | ~16GB |
| Quantize AWQ | 30-60 min | ~20GB |
| **Total** | **~4-6 hours** | **Fits on 3090** |


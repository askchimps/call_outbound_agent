# Deployment Guide

## Part 1: Setup vLLM on Vast.ai

### Step 1: Rent a GPU Instance
1. Go to [vast.ai](https://vast.ai) and create an account
2. Navigate to **Search** → **Templates**
3. Search for **vLLM** template
4. Select **RTX 3090 (24GB)** - sufficient for Qwen2.5-7B-Instruct-AWQ
5. Look for instance in **India** region for lowest latency
6. Click **Rent** and wait for the instance to start

### Step 2: Connect to the Instance
```bash
ssh -p <PORT> root@<VAST_AI_IP>
```

### Step 3: Stop the Default Model
The instance comes with a pre-loaded model. Stop it first:
```bash
# Find and kill the running vLLM process
pkill -f vllm

# Or find the process manually
ps aux | grep vllm
kill <PID>
```

### Step 4: Install Process Manager (tmux)
We'll use `tmux` to keep vLLM running in the background (similar to pm2 for Node.js):
```bash
apt update && apt install -y tmux
```

### Step 5: Deploy Qwen Model with tmux
```bash
# Create a new tmux session named 'vllm'
tmux new -s vllm

# Inside tmux, start vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

# Detach from tmux: Press Ctrl+B, then D
```

### Step 6: tmux Commands Reference
```bash
# List all sessions
tmux ls

# Attach to vllm session (view logs)
tmux attach -t vllm

# Detach from session (keep running): Ctrl+B, then D

# Kill the session (stop vLLM)
tmux kill-session -t vllm

# Scroll through logs: Ctrl+B, then [ (use arrow keys, q to exit)
```

### Step 7: Verify vLLM is Running
```bash
curl http://localhost:8000/v1/models
```

Expected response:
```json
{"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", ...}]}
```

### Step 8: Note Your Vast.ai Public IP
Find your instance's public IP and port from the Vast.ai dashboard. You'll need this for `LLM_BASE_URL`.

### Alternative: Using systemd (if available)
Create a systemd service for auto-restart:
```bash
cat > /etc/systemd/system/vllm.service << 'EOF'
[Unit]
Description=vLLM Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192
Restart=always
RestartSec=10

[Service]
StandardOutput=append:/var/log/vllm.log
StandardError=append:/var/log/vllm.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable vllm
systemctl start vllm

# Check status
systemctl status vllm

# View logs
tail -f /var/log/vllm.log
journalctl -u vllm -f
```

---

## Part 2: Deploy Agent on AWS EC2

### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 22.04 LTS**
3. Instance type: **t3.medium** or larger
4. Configure security group:
   - SSH (22) - Your IP
   - HTTP (8000) - Anywhere (for API)
5. Launch and download the key pair

### Step 2: Connect to EC2
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 3: Install Docker
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 4: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/outbound-caller-python.git
cd outbound-caller-python
```

### Step 5: Configure Environment
```bash
cp .env.example .env.local
nano .env.local
```

Set the following values:
```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=not-needed
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_BASE_URL=http://<VAST_AI_IP>:<PORT>/v1
SIP_OUTBOUND_TRUNK_ID=your_trunk_id
ELEVEN_API_KEY=your_elevenlabs_key
TTS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
TTS_MODEL_ID=eleven_flash_v2_5
```

### Step 6: Create baseprompt.txt
```bash
nano baseprompt.txt
```
Paste your agent's system prompt.

### Step 7: Build and Run
```bash
docker-compose up -d --build
```

### Step 8: Check Logs
```bash
docker-compose logs -f
```

---

## Part 3: Test the API

### Health Check
```bash
curl http://<EC2_PUBLIC_IP>:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### Make an Outbound Call
```bash
curl -X POST http://<EC2_PUBLIC_IP>:8000/dispatch \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+919876543210"}'
```

Expected response:
```json
{
  "dispatch_id": "...",
  "room_name": "call-...",
  "phone_number": "+919876543210",
  "status": "dispatched"
}
```

---

## Troubleshooting

### vLLM Issues
- **Out of memory**: Reduce `--max-model-len` or use smaller model
- **Model not found**: Check Hugging Face access for gated models
- **Check vLLM logs**: `tmux attach -t vllm` or `tail -f /var/log/vllm.log`

### Agent Issues
- **Connection refused**: Check security groups allow port 8000
- **LLM timeout**: Verify Vast.ai instance is running and accessible

### Check Container Status
```bash
docker-compose ps
docker-compose logs outbound-caller
```

### Useful Commands
```bash
# Restart agent
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# Stop everything
docker-compose down

# View real-time logs
docker-compose logs -f
```

---

## Part 4: Fine-Tuning Guide (For Later)

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


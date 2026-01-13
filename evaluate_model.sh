export PROJECT_PATH=/root/autodl-tmp/HiAgent
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
# 自动启动 SSH 隧道
chmod +x start_tunnel.sh
./start_tunnel.sh

MODEL="mistral-7b"
export STEP=50
AGENT="ContextEfficientAgentV2"
export EVALTASK="blocksworld"
# Use conda environment
/root/autodl-tmp/conda/envs/env/bin/python agentboard/eval_main.py \
    --cfg-path eval_configs/hiagent/blocksworld.yaml\
    --tasks pddl \
    --model $MODEL  \
    --log_path ./logs/hiagent/analysis/"${AGENT}_${MODEL}_${STEP}"   \
    --project_name hiagent-evaluate \
    --baseline_dir ./data/baseline_results  \
    --max_num_steps $STEP     \
    --memory_size   100  \
    --agent $AGENT \
    # --wandb
#!/bin/bash
export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH=${1:-"Qwen/Qwen3-32B"}

cd /home/alignment-without-rewards/verl-0.7.0/recipe/gkd/teacher

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "Wait $server server ready at $ip:$port..."
        result=$(echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l)
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

nohup python proxy.py > proxy.log 2>&1 &
wait_server_ready proxy localhost $PROXY_BACKEND_PORT
echo "Teacher proxy is ready."

nohup python worker.py --backend $BACKEND --tp-size 1 --n-logprobs 20 --ckpt-path "$CKPT_PATH" --gpu-memory-utilization 0.36 > worker.log 2>&1 &
echo "Started teacher worker."
echo "Teacher server initialization complete! You can view the server logs at:"
echo "  /home/alignment-without-rewards/verl-0.7.0/recipe/gkd/teacher/worker.log"

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH=${1:-"Qwen/Qwen3-32B"}

# Change to the script's directory so proxy.py can be found
cd "$(dirname "$0")" || exit 1

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        python -c "import socket; s = socket.socket(); s.settimeout(1); s.connect(('$ip', int('$port'))); s.close()" 2>/dev/null
        if [ $? -eq 0 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --tp-size 1 --n-logprobs 20 --ckpt-path "$CKPT_PATH" --gpu-memory-utilization 0.36 &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"
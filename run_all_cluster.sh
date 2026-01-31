#!/bin/bash

set -euo pipefail

print_usage() {
    echo "用法: bash run_all_cluster.sh [--out-dir DIR] [--poll-interval SEC] [--no-wait] [--python-bin /path/to/python]"
    echo "                       [--check-only]"
    echo "  默认模式：master-worker（多节点/多 GPU 动态抢占队列并行跑不同 case）"
    echo "  --out-dir        本地汇总输出目录（默认：<repo>/cluster_reports_<timestamp>）"
    echo "  --poll-interval  轮询远端进度间隔秒数（默认：30）"
    echo "  --no-wait        只触发远端 worker 运行，不等待、不汇总"
    echo "  --python-bin     指定远端执行用的 Python（建议指向 conda env 的 python）"
    echo "  --check-only     只做 SSH/Python/torch 预检，不启动任务"
}

OUT_DIR=""
POLL_INTERVAL=30
WAIT_FOR_COMPLETION=1
REMOTE_PYTHON_BIN=""
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --no-wait)
            WAIT_FOR_COMPLETION=0
            shift 1
            ;;
        --python-bin)
            REMOTE_PYTHON_BIN="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ================================================================
# - 优先使用 IB 内网地址进行 SSH/通信
# - 使用域名作为 label
# ================================================================

HOST_LABELS=("icarus0.acsalab.com" "icarus1.acsalab.com" "icarus2.acsalab.com" "icarus3.acsalab.com")
HOST_ADDRS=("10.1.26.220" "10.1.26.221" "10.1.26.222" "10.1.26.223")

# SSH 选项：BatchMode 避免卡住；StrictHostKeyChecking=accept-new 首次连接自动记录指纹
# 加上超时与 keepalive，避免某个节点网络问题导致脚本卡住。
SSH_OPTS=(
    "-o" "BatchMode=yes"
    "-o" "StrictHostKeyChecking=accept-new"
    "-o" "ConnectTimeout=5"
    "-o" "ServerAliveInterval=30"
    "-o" "ServerAliveCountMax=3"
)

# 远端仓库路径
REMOTE_REPO_DIR="/home/chenyq/code/unifolm-world-model-action"

# 每台机器参与的 GPU 列表（传给 run_all_parallel.sh 的 --gpus）
REMOTE_GPU_IDS="0,1"

# SSH 启动不会继承你当前 shell 里激活的 conda 环境，因此必须显式指定远端 python。
# 默认策略：优先使用 NFS 共享的 worldmodel 环境 python；否则再尝试当前 CONDA_PREFIX；最后退回 python3。
if [[ -z "$REMOTE_PYTHON_BIN" ]]; then
    if [[ -x "/home/chenyq/.conda/envs/worldmodel/bin/python" ]]; then
        REMOTE_PYTHON_BIN="/home/chenyq/.conda/envs/worldmodel/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
        REMOTE_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        REMOTE_PYTHON_BIN="python3"
    fi
fi

echo ">>> 将使用远端 Python: $REMOTE_PYTHON_BIN"

remote_python_ok() {
    local host_addr="$1"
    # 最小自检：python 存在且能 import torch（cuda 可用性不强制为 true）
    ssh "${SSH_OPTS[@]}" "$host_addr" "$REMOTE_PYTHON_BIN -c 'import torch; import sys; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'"
}

echo ">>> 预检各节点 Python/torch 环境..."
for host_addr in "${HOST_ADDRS[@]}"; do
    echo "[$host_addr]"
    if ! remote_python_ok "$host_addr"; then
        echo "[错误] 节点 $host_addr 无法用 $REMOTE_PYTHON_BIN import torch（或 python 不存在）。"
        echo "建议在该节点上执行："
        echo "  ssh $host_addr '$REMOTE_PYTHON_BIN -c \"import torch; print(torch.__version__)\"'"
        echo "该节点的错误输出（便于定位）："
        ssh "${SSH_OPTS[@]}" "$host_addr" "$REMOTE_PYTHON_BIN -c 'import torch; print(torch.__version__)'" 2>&1 || true
        echo "解决方式："
        echo "  1) 确认 worldmodel 环境在 NFS 共享路径上，且该 python 路径在所有节点都存在"
        echo "  2) 或者显式指定：bash run_all_cluster.sh --python-bin /path/to/conda/env/bin/python"
        exit 2
    fi
done

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ">>> 预检完成：所有节点 OK（--check-only 退出）"
    exit 0
fi

SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

ALL_CASES=()
for s in "${SCENARIOS[@]}"; do
    for c in "${CASES[@]}"; do
        ALL_CASES+=("$s:$c")
    done
done

NUM_HOSTS=${#HOST_ADDRS[@]}
if [[ $NUM_HOSTS -lt 1 ]]; then
    echo "HOSTS 为空，无法分发"
    exit 1
fi

if [[ ${#HOST_LABELS[@]} -ne ${#HOST_ADDRS[@]} ]]; then
    echo "HOST_LABELS 与 HOST_ADDRS 数量不一致"
    exit 1
fi

TMP_DIR=$(mktemp -d)

echo ">>> 将 ${#ALL_CASES[@]} 个任务分发到 ${NUM_HOSTS} 台机器"

if [[ -z "$OUT_DIR" ]]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    OUT_DIR="$SCRIPT_DIR/cluster_reports_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

# 保存一份 host 映射，便于排查问题
HOSTS_MAP_FILE="$OUT_DIR/hosts.map"
: > "$HOSTS_MAP_FILE"

LOCAL_MERGED="$OUT_DIR/cluster_benchmark_report.log"
LOCAL_TABLE="$OUT_DIR/cluster_benchmark_table.tsv"

start_master_worker() {
    echo ">>> 使用 master-worker 动态队列分配任务"

    local queue_file="$OUT_DIR/cases.queue"
    local queue_lock_dir="$OUT_DIR/.queue_lock"
    local result_tsv="$OUT_DIR/results.tsv"
    local result_lock_dir="$OUT_DIR/.result_lock"
    local log_dir="$OUT_DIR/logs"
    mkdir -p "$log_dir"
    rm -f "$queue_file" "$result_tsv" 2>/dev/null || true
    rmdir "$queue_lock_dir" 2>/dev/null || true
    rmdir "$result_lock_dir" 2>/dev/null || true

    printf "%s\n" "${ALL_CASES[@]}" > "$queue_file"
    echo -e "host\tgpu\tcase\ttime\tpsnr\texit_code" > "$result_tsv"

    # 为每个节点每张 GPU 启动一个 worker（并发拉任务）
    IFS=',' read -r -a GPU_LIST <<< "$REMOTE_GPU_IDS"
    for idx in "${!HOST_ADDRS[@]}"; do
        host_label="${HOST_LABELS[$idx]}"
        host_addr="${HOST_ADDRS[$idx]}"

        for gpu in "${GPU_LIST[@]}"; do
            remote_log="/tmp/wma_worker_${host_label}_gpu${gpu}_$$.log"
            ssh "${SSH_OPTS[@]}" "$host_addr" "cd '$REMOTE_REPO_DIR' && nohup bash cluster_worker.sh \
                --repo-dir '$REMOTE_REPO_DIR' \
                --host-label '$host_label' \
                --gpu '$gpu' \
                --python-bin '$REMOTE_PYTHON_BIN' \
                --queue-file '$queue_file' \
                --queue-lock-dir '$queue_lock_dir' \
                --result-tsv '$result_tsv' \
                --result-lock-dir '$result_lock_dir' \
                --log-dir '$log_dir' \
                > '$remote_log' 2>&1 & echo \$!" \
                > "$TMP_DIR/pid_${host_label}_gpu${gpu}.txt"

            pid=$(cat "$TMP_DIR/pid_${host_label}_gpu${gpu}.txt" | tr -d '\r\n' | awk '{print $1}')
            echo "$host_label|$host_addr|$pid|$remote_log|$result_tsv" >> "$HOSTS_MAP_FILE"
            echo "[$host_label @ $host_addr] worker gpu=$gpu pid=$pid log=$remote_log"
        done
    done
}

start_master_worker

echo ""
echo "[完成] 已在所有机器上触发运行。"
echo "你可以用下面命令查看某台机器 worker 日志："
echo "  ssh 10.1.26.221 'ls -1t /tmp/wma_worker_* | head'"
echo "  ssh 10.1.26.221 'tail -f /tmp/wma_worker_icarus1.acsalab.com_gpu0_*.log'"
echo "或者在共享目录看全局结果："
echo "  tail -f $OUT_DIR/results.tsv"

if [[ $WAIT_FOR_COMPLETION -eq 0 ]]; then
    echo ""
    echo "[跳过] --no-wait 已指定，不等待/不拉取/不汇总。"
    rm -rf "$TMP_DIR" 2>/dev/null || true
    exit 0
fi

echo "" | tee "$LOCAL_MERGED" > /dev/null
echo "================ ASC26 四机汇总报告 ================" | tee -a "$LOCAL_MERGED" > /dev/null
echo "开始时间: $(date)" | tee -a "$LOCAL_MERGED" > /dev/null
echo "远端仓库: $REMOTE_REPO_DIR" | tee -a "$LOCAL_MERGED" > /dev/null
echo "GPUs(每机): $REMOTE_GPU_IDS" | tee -a "$LOCAL_MERGED" > /dev/null
echo "Python: $REMOTE_PYTHON_BIN" | tee -a "$LOCAL_MERGED" > /dev/null
echo "Mode: master-worker" | tee -a "$LOCAL_MERGED" > /dev/null
echo "----------------------------------------------------" | tee -a "$LOCAL_MERGED" > /dev/null

is_pid_running_remote() {
    local host_addr="$1"
    local pid="$2"
    ssh "${SSH_OPTS[@]}" "$host_addr" "kill -0 '$pid' >/dev/null 2>&1" >/dev/null 2>&1
}

echo ">>> 等待所有机器任务完成（轮询间隔 ${POLL_INTERVAL}s）..."
while true; do
    running=0
    while IFS='|' read -r host_label host_addr pid remote_log remote_report; do
        if is_pid_running_remote "$host_addr" "$pid"; then
            running=$((running + 1))
        fi
    done < "$HOSTS_MAP_FILE"

    if [[ $running -eq 0 ]]; then
        break
    fi

    total_workers=$(grep -c '|' "$HOSTS_MAP_FILE" 2>/dev/null || echo 0)
    echo "仍在运行的 worker 数: $running / $total_workers (next check in ${POLL_INTERVAL}s)"
    sleep "$POLL_INTERVAL"
done

echo ">>> 远端任务已全部结束，开始拉取报告并汇总..."

RESULT_TSV="$OUT_DIR/results.tsv"
if [[ -f "$RESULT_TSV" ]]; then
    echo -e "host\tcase\ttime\tpsnr\tdevice\texit_code" > "$LOCAL_TABLE"
    tail -n +2 "$RESULT_TSV" \
        | grep -v "__worker_" \
        | awk -F'\t' '{
            host=$1; gpu=$2; case=$3; time=$4; psnr=$5; exit_code=$6;
            print host"\t"case"\t"time"\t"psnr"\tGPU "gpu"\t"exit_code;
        }' >> "$LOCAL_TABLE" || true
else
    echo "[警告] 未找到 results.tsv：$RESULT_TSV" | tee -a "$LOCAL_MERGED" > /dev/null
    echo -e "host\tcase\ttime\tpsnr\tdevice\texit_code" > "$LOCAL_TABLE"
fi

echo "" | tee -a "$LOCAL_MERGED" > /dev/null
echo "================ 汇总表 (TSV) ================" | tee -a "$LOCAL_MERGED" > /dev/null
echo "文件: $LOCAL_TABLE" | tee -a "$LOCAL_MERGED" > /dev/null

# 输出一个按 case 排序的可读表到 merged 报告尾部
echo "host | case | time | psnr | device | exit_code" | tee -a "$LOCAL_MERGED" > /dev/null
echo "----------------------------------------------------" | tee -a "$LOCAL_MERGED" > /dev/null
tail -n +2 "$LOCAL_TABLE" | sort -t$'\t' -k2,2 | awk -F'\t' '{printf "%s | %s | %s | %s | %s | %s\n", $1,$2,$3,$4,$5,$6}' \
    | tee -a "$LOCAL_MERGED" > /dev/null

echo "----------------------------------------------------" | tee -a "$LOCAL_MERGED" > /dev/null
echo "完成时间: $(date)" | tee -a "$LOCAL_MERGED" > /dev/null
echo "输出目录: $OUT_DIR" | tee -a "$LOCAL_MERGED" > /dev/null

rm -rf "$TMP_DIR" 2>/dev/null || true

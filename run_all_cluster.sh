set -euo pipefail

print_usage() {
    echo "用法: bash run_all_cluster.sh [--out-dir DIR] [--poll-interval SEC] [--no-wait] [--python-bin /path/to/python]"
    echo "  --out-dir        本地汇总输出目录（默认：<repo>/cluster_reports_<timestamp>）"
    echo "  --poll-interval  轮询远端进度间隔秒数（默认：30）"
    echo "  --no-wait        只触发远端运行，不等待、不拉取、不汇总"
    echo "  --python-bin     指定远端执行用的 Python（建议指向 conda env 的 python）"
}

OUT_DIR=""
POLL_INTERVAL=30
WAIT_FOR_COMPLETION=1
REMOTE_PYTHON_BIN=""

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
SSH_OPTS=("-o" "BatchMode=yes" "-o" "StrictHostKeyChecking=accept-new")

# 远端仓库路径
REMOTE_REPO_DIR="/home/chenyq/code/unifolm-world-model-action"

# 每台机器参与的 GPU 列表（传给 run_all_parallel.sh 的 --gpus）
REMOTE_GPU_IDS="0,1"

# SSH 启动不会继承你当前 shell 里激活的 conda 环境，因此必须显式指定远端 python。
# 默认策略：如果你当前是在 conda env（比如 worldmodel）里运行该脚本，则用 $CONDA_PREFIX/bin/python。
if [[ -z "$REMOTE_PYTHON_BIN" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
        REMOTE_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        REMOTE_PYTHON_BIN="python3"
    fi
fi

echo ">>> 将使用远端 Python: $REMOTE_PYTHON_BIN"

remote_python_ok() {
    local host_addr="$1"
    # 依赖 torch 的最小自检 + CUDA 可用性打印（不要求 cuda=true，但至少 torch 能 import）
    ssh "${SSH_OPTS[@]}" "$host_addr" "$REMOTE_PYTHON_BIN - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
except Exception as e:
    print('TORCH_IMPORT_FAILED:', repr(e))
    sys.exit(2)
PY" >/dev/null 2>&1
}

echo ">>> 预检各节点 Python/torch 环境..."
for host_addr in "${HOST_ADDRS[@]}"; do
    if ! remote_python_ok "$host_addr"; then
        echo "[错误] 节点 $host_addr 无法用 $REMOTE_PYTHON_BIN import torch（或 python 不存在）。"
        echo "解决方式："
        echo "  1) 确认 worldmodel 环境在 NFS 共享路径上，且该 python 路径在所有节点都存在"
        echo "  2) 或者显式指定：bash run_all_cluster.sh --python-bin /path/to/conda/env/bin/python"
        exit 2
    fi
done

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

# 为每台机器准备一个 case list（round-robin）
TMP_DIR=$(mktemp -d)
LIST_FILES=()
for label in "${HOST_LABELS[@]}"; do
    safe_label=$(echo "$label" | sed 's/[^a-zA-Z0-9._-]/_/g')
    f="$TMP_DIR/cases_${safe_label}.txt"
    : > "$f"
    LIST_FILES+=("$f")
done

for i in "${!ALL_CASES[@]}"; do
    host_idx=$((i % NUM_HOSTS))
    echo "${ALL_CASES[$i]}" >> "${LIST_FILES[$host_idx]}"
done

echo ">>> 将 ${#ALL_CASES[@]} 个任务分发到 ${NUM_HOSTS} 台机器"

deploy_and_run_on_host() {
    local host_label="$1"
    local host_addr="$2"
    local list_file="$3"

    local safe_label
    safe_label=$(echo "$host_label" | sed 's/[^a-zA-Z0-9._-]/_/g')

    local remote_list="/tmp/wma_cases_${safe_label}_$$.txt"
    local remote_log="/tmp/wma_run_${safe_label}_$$.log"
    local remote_report="$REMOTE_REPO_DIR/benchmark_report_${safe_label}.log"

    echo "[$host_label @ $host_addr] 上传 case list 到 $remote_list 并启动运行"

    # 把 case list 通过 stdin 写到远端文件，避免依赖 scp/rsync
    ssh "${SSH_OPTS[@]}" "$host_addr" "cat > '$remote_list'" < "$list_file"

    # 后台运行
    ssh "${SSH_OPTS[@]}" "$host_addr" "cd '$REMOTE_REPO_DIR' && PYTHON_BIN='$REMOTE_PYTHON_BIN' nohup bash run_all_parallel.sh --gpus '$REMOTE_GPU_IDS' --case-list '$remote_list' --report '$remote_report' > '$remote_log' 2>&1 & echo \$!" \
        > "$TMP_DIR/pid_${safe_label}.txt"

    local pid
    pid=$(cat "$TMP_DIR/pid_${safe_label}.txt" | tr -d '\r\n')
    echo "[$host_label] 已启动 PID=$pid"
    echo "[$host_label] 日志: $remote_log"
    echo "[$host_label] 报告: $remote_report"

    # 记录映射，供后续等待/拉取
    echo "$host_label|$host_addr|$pid|$remote_log|$remote_report" >> "$TMP_DIR/hosts.map"
}

for idx in "${!HOST_ADDRS[@]}"; do
    deploy_and_run_on_host "${HOST_LABELS[$idx]}" "${HOST_ADDRS[$idx]}" "${LIST_FILES[$idx]}"
done

echo ""
echo "[完成] 已在所有机器上触发运行。"
echo "你可以用下面命令查看某台机器进度："
echo "  ssh 10.1.26.221 'tail -f /tmp/wma_run_icarus1.acsalab.com_*.log'"
echo "或者查看远端报告："
echo "  ssh 10.1.26.221 'ls -1 $REMOTE_REPO_DIR/benchmark_report_*.log'"

if [[ $WAIT_FOR_COMPLETION -eq 0 ]]; then
    echo ""
    echo "[跳过] --no-wait 已指定，不等待/不拉取/不汇总。"
    rm -rf "$TMP_DIR" 2>/dev/null || true
    exit 0
fi

if [[ -z "$OUT_DIR" ]]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    OUT_DIR="$SCRIPT_DIR/cluster_reports_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"

# 保存一份 host 映射，便于排查问题
cp "$TMP_DIR/hosts.map" "$OUT_DIR/hosts.map" 2>/dev/null || true

LOCAL_MERGED="$OUT_DIR/cluster_benchmark_report.log"
LOCAL_TABLE="$OUT_DIR/cluster_benchmark_table.tsv"

echo "" | tee "$LOCAL_MERGED" > /dev/null
echo "================ ASC26 四机汇总报告 ================" | tee -a "$LOCAL_MERGED" > /dev/null
echo "开始时间: $(date)" | tee -a "$LOCAL_MERGED" > /dev/null
echo "远端仓库: $REMOTE_REPO_DIR" | tee -a "$LOCAL_MERGED" > /dev/null
echo "GPUs(每机): $REMOTE_GPU_IDS" | tee -a "$LOCAL_MERGED" > /dev/null
echo "Python: $REMOTE_PYTHON_BIN" | tee -a "$LOCAL_MERGED" > /dev/null
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
    done < "$TMP_DIR/hosts.map"

    if [[ $running -eq 0 ]]; then
        break
    fi

    echo "仍在运行的节点数: $running / $NUM_HOSTS (next check in ${POLL_INTERVAL}s)"
    sleep "$POLL_INTERVAL"
done

echo ">>> 远端任务已全部结束，开始拉取报告并汇总..."

# 表头：host\tcase\ttime\tpsnr\tdevice
echo -e "host\tcase\ttime\tpsnr\tdevice" > "$LOCAL_TABLE"

while IFS='|' read -r host_label host_addr pid remote_log remote_report; do
    safe_label=$(echo "$host_label" | sed 's/[^a-zA-Z0-9._-]/_/g')
    local_host_report="$OUT_DIR/benchmark_report_${safe_label}.log"
    local_host_log="$OUT_DIR/run_${safe_label}.log"

    # 拉远端 stdout/stderr 日志
    ssh "${SSH_OPTS[@]}" "$host_addr" "cat '$remote_log'" > "$local_host_log" 2>/dev/null || true

    # 报告优先从共享 /home(NFS) 直接读取；若不存在再尝试 SSH 拉取
    if [[ -f "$remote_report" ]]; then
        cp "$remote_report" "$local_host_report" 2>/dev/null || true
    elif ssh "${SSH_OPTS[@]}" "$host_addr" "test -f '$remote_report'" >/dev/null 2>&1; then
        ssh "${SSH_OPTS[@]}" "$host_addr" "cat '$remote_report'" > "$local_host_report" 2>/dev/null || true
    else
        echo "[$host_label] 报告不存在: $remote_report" | tee -a "$LOCAL_MERGED" > /dev/null
        continue
    fi

    echo "" | tee -a "$LOCAL_MERGED" > /dev/null
    echo "---------------- $host_label ($host_addr) ----------------" | tee -a "$LOCAL_MERGED" > /dev/null
    tail -n +1 "$local_host_report" | tee -a "$LOCAL_MERGED" > /dev/null

    # 抽取每行结果并转换成 TSV 追加到总表
    # 原行格式：CASE_NAME | TIME_DESC | PSNR | GPU X
    grep -E "^[a-zA-Z0-9._-]+-case[0-9]+[[:space:]]*\|" "$local_host_report" \
        | awk -F'\|' -v host="$host_label" '{
            gsub(/^[ \t]+|[ \t]+$/, "", $1);
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            gsub(/^[ \t]+|[ \t]+$/, "", $3);
            gsub(/^[ \t]+|[ \t]+$/, "", $4);
            print host"\t"$1"\t"$2"\t"$3"\t"$4;
        }' >> "$LOCAL_TABLE" || true

done < "$TMP_DIR/hosts.map"

echo "" | tee -a "$LOCAL_MERGED" > /dev/null
echo "================ 汇总表 (TSV) ================" | tee -a "$LOCAL_MERGED" > /dev/null
echo "文件: $LOCAL_TABLE" | tee -a "$LOCAL_MERGED" > /dev/null

# 输出一个按 case 排序的可读表到 merged 报告尾部
echo "host | case | time | psnr | device" | tee -a "$LOCAL_MERGED" > /dev/null
echo "----------------------------------------------------" | tee -a "$LOCAL_MERGED" > /dev/null
tail -n +2 "$LOCAL_TABLE" | sort -t$'\t' -k2,2 | awk -F'\t' '{printf "%s | %s | %s | %s | %s\n", $1,$2,$3,$4,$5}' \
    | tee -a "$LOCAL_MERGED" > /dev/null

echo "----------------------------------------------------" | tee -a "$LOCAL_MERGED" > /dev/null
echo "完成时间: $(date)" | tee -a "$LOCAL_MERGED" > /dev/null
echo "输出目录: $OUT_DIR" | tee -a "$LOCAL_MERGED" > /dev/null

rm -rf "$TMP_DIR" 2>/dev/null || true

#!/bin/bash

# =================================================================
# ASC26 双卡并行基准测试脚本 (Multi-GPU Parallel Baseline)
# 用于将多个 Case 分配到多张 GPU 上并行运行
# =================================================================

print_usage() {
    echo "用法: bash run_all_parallel.sh [--gpus 0,1] [--case-list /path/to/list.txt]"
    echo "                       [--report /path/to/benchmark_report.log]"
    echo "环境变量：PYTHON_BIN=/path/to/python (默认 python3)"
    echo "  --gpus       逗号分隔的 GPU 列表，默认: 0,1"
    echo "  --case-list  指定只运行的 Case 列表文件，每行一个: <scenario>:<caseX>"
    echo "  --report     报告输出文件路径，默认: <repo>/benchmark_report.log"
}

GPU_IDS="0,1"
CASE_LIST_FILE=""
REPORT_FILE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --case-list)
            CASE_LIST_FILE="$2"
            shift 2
            ;;
        --report)
            REPORT_FILE_OVERRIDE="$2"
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

SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

# 获取所有 Case 的列表
ALL_CASES=()
if [[ -n "$CASE_LIST_FILE" ]]; then
    if [[ ! -f "$CASE_LIST_FILE" ]]; then
        echo "找不到 --case-list 文件: $CASE_LIST_FILE"
        exit 1
    fi
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [[ -z "$line" ]] && continue
        ALL_CASES+=("$line")
    done < "$CASE_LIST_FILE"
else
    for s in "${SCENARIOS[@]}"; do
        for c in "${CASES[@]}"; do
            ALL_CASES+=("$s:$c")
        done
    done
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"
REPORT_FILE="$ROOT_DIR/benchmark_report.log"
if [[ -n "$REPORT_FILE_OVERRIDE" ]]; then
    REPORT_FILE="$REPORT_FILE_OVERRIDE"
fi
PSNR_SCRIPT="$ROOT_DIR/psnr_score_for_challenge.py"
PYTHON_BIN=${PYTHON_BIN:-python3}

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS=${#GPU_LIST[@]}
if [[ $NUM_GPUS -lt 1 ]]; then
    echo "--gpus 不能为空"
    exit 1
fi

if [[ ! -f "$PSNR_SCRIPT" ]]; then
    echo "找不到 PSNR 脚本: $PSNR_SCRIPT"
    echo "请确认仓库根目录下存在 psnr_score_for_challenge.py"
    exit 1
fi

if [[ "$PYTHON_BIN" == /* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "PYTHON_BIN 不可执行: $PYTHON_BIN"
        exit 1
    fi
else
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "找不到 PYTHON_BIN: $PYTHON_BIN"
        exit 1
    fi
fi

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "Python 环境缺少 torch：$PYTHON_BIN"
    echo "请先在该机器上安装依赖，或设置 PYTHON_BIN 指向带 torch 的 venv/conda。"
    echo "示例：PYTHON_BIN=/home/chenyq/venv/unifolm/bin/python"
    exit 2
fi

# 初始化报告
echo "================ ASC26 双卡并行基准测试报告 ================" > "$REPORT_FILE"
echo "开始时间: $(date)" >> "$REPORT_FILE"
echo "-------------------------------------------------------------------" >> "$REPORT_FILE"

# 并行执行函数
run_case_on_gpu() {
    local gpu_id=$1
    local scenario_case=$2
    
    # 拆分场景和 Case
    local scenario=$(echo $scenario_case | cut -d':' -f1)
    local case_id=$(echo $scenario_case | cut -d':' -f2)
    local CASE_NAME="${scenario}-${case_id}"
    local CASE_DIR="$ROOT_DIR/$scenario/$case_id"
    local CKPT_PATH="$ROOT_DIR/ckpts/unifolm_wma_dual.ckpt"
    local CONFIG_PATH="$ROOT_DIR/configs/inference/world_model_interaction.yaml"
    local SAVE_DIR="$CASE_DIR/output"
    local PROMPT_DIR="$CASE_DIR/world_model_interaction_prompts"

    local frame_stride=4
    local n_iter=11
    case "$scenario" in
        unitree_z1_stackbox)
            frame_stride=4; n_iter=12 ;;
        unitree_z1_dual_arm_stackbox)
            frame_stride=4; n_iter=7 ;;
        unitree_z1_dual_arm_stackbox_v2)
            frame_stride=4; n_iter=11 ;;
        unitree_z1_dual_arm_cleanup_pencils)
            frame_stride=4; n_iter=8 ;;
        unitree_g1_pack_camera)
            frame_stride=6; n_iter=11 ;;
        *)
            frame_stride=4; n_iter=11 ;;
    esac

    echo "[GPU $gpu_id] 正在启动: $CASE_NAME"

    # 执行推理 (强制指定 CUDA_VISIBLE_DEVICES)
    START_TIME=$(date +%s)
    
    cd "$ROOT_DIR"

    if [[ ! -f "$CKPT_PATH" ]]; then
        echo "缺少 ckpt: $CKPT_PATH" > "$CASE_DIR/parallel_run.log"
    elif [[ ! -f "$CONFIG_PATH" ]]; then
        echo "缺少 config: $CONFIG_PATH" > "$CASE_DIR/parallel_run.log"
    elif [[ ! -d "$PROMPT_DIR" ]]; then
        echo "缺少 prompt_dir: $PROMPT_DIR" > "$CASE_DIR/parallel_run.log"
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" scripts/evaluation/world_model_interaction.py \
            --seed 123 \
            --ckpt_path "$CKPT_PATH" \
            --config "$CONFIG_PATH" \
            --savedir "$SAVE_DIR" \
            --bs 1 --height 320 --width 512 \
            --unconditional_guidance_scale 1.0 \
            --ddim_steps 50 \
            --ddim_eta 1.0 \
            --prompt_dir "$PROMPT_DIR" \
            --dataset "$scenario" \
            --video_length 16 \
            --frame_stride "$frame_stride" \
            --n_action_steps 16 \
            --exe_steps 16 \
            --n_iter "$n_iter" \
            --timestep_spacing 'uniform_trailing' \
            --guidance_rescale 0.7 \
            --perframe_ae \
            > "$CASE_DIR/parallel_run.log" 2>&1
    fi

    local exit_code=$?
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    TIME_DESC="$(($ELAPSED / 60))m $(($ELAPSED % 60))s"

    # 计算 PSNR
    local PRED_VIDEO=""
    shopt -s nullglob
    local preds=("$CASE_DIR"/output/inference/*.mp4)
    shopt -u nullglob
    if [[ ${#preds[@]} -gt 0 ]]; then
        PRED_VIDEO="${preds[0]}"
    fi
    GT_VIDEO="$CASE_DIR/${scenario}_${case_id}.mp4"
    SCORE_JSON="$CASE_DIR/psnr_score.json"
    
    PSNR_VAL="N/A"
    if [[ $exit_code -ne 0 ]]; then
        PSNR_VAL="ERR(exit=$exit_code)"
    elif [ -f "$PRED_VIDEO" ] && [ -f "$GT_VIDEO" ]; then
        "$PYTHON_BIN" "$PSNR_SCRIPT" --gt_video="$GT_VIDEO" --pred_video="$PRED_VIDEO" --output_file="$SCORE_JSON" > /dev/null 2>&1
        PSNR_VAL=$($PYTHON_BIN -c "import json; print(json.load(open('$SCORE_JSON'))['psnr'])" 2>/dev/null || echo "Err")
    fi

    # 写入临时结果文件（避免并发写入冲突）
    {
        flock -x 9
        echo "$CASE_NAME | $TIME_DESC | $PSNR_VAL | GPU $gpu_id" >> "${REPORT_FILE}.tmp"
    } 9>>"${REPORT_FILE}.tmp.lock"
    echo "[GPU $gpu_id] 完成: $CASE_NAME (耗时: $TIME_DESC)"
}

# 任务队列：确保同一时间每张 GPU 只跑一个 Case（避免一开始就把 20 个都丢到后台）
QUEUE_DIR=$(mktemp -d)
QUEUE_FILE="$QUEUE_DIR/cases.queue"
QUEUE_LOCK="$QUEUE_DIR/cases.lock"
printf "%s\n" "${ALL_CASES[@]}" > "$QUEUE_FILE"

pop_job() {
    local job=""
    exec 200>"$QUEUE_LOCK"
    flock -x 200
    job=$(head -n 1 "$QUEUE_FILE")
    if [[ -n "$job" ]]; then
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" 2>/dev/null || true
        mv "$QUEUE_FILE.tmp" "$QUEUE_FILE" 2>/dev/null || true
    fi
    flock -u 200
    exec 200>&-
    echo "$job"
}

worker() {
    local gpu_id="$1"
    while true; do
        local job
        job=$(pop_job)
        [[ -z "$job" ]] && break
        run_case_on_gpu "$gpu_id" "$job"
    done
}

# 记录总起始时间
TOTAL_START=$(date +%s)

# 分配原则：
echo ">>> 开始并行推理 (GPUs: $GPU_IDS)，总任务数: ${#ALL_CASES[@]}"

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
    sleep 1
done

# 等待所有后台进程结束
wait

rm -rf "$QUEUE_DIR" 2>/dev/null || true

# 汇总结果
echo "场景-Case                                | 耗时           | PSNR     | 运行设备" >> "$REPORT_FILE"
echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
sort "${REPORT_FILE}.tmp" >> "$REPORT_FILE"
rm "${REPORT_FILE}.tmp"

rm -f "${REPORT_FILE}.tmp.lock" 2>/dev/null || true

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_DESC="$(($TOTAL_DIFF / 3600))h $((($TOTAL_DIFF % 3600) / 60))m $(($TOTAL_DIFF % 60))s"

echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
echo "双卡总并发执行效率时长: $TOTAL_DESC" >> "$REPORT_FILE"
echo "完成时间: $(date)" >> "$REPORT_FILE"

echo -e "\n[!!!] 双卡并行运行完毕！"
echo "[!!!] 请查看报告: $REPORT_FILE"

#!/bin/bash

# ================================================================
# Cluster worker: pull-based master-worker over shared /home (NFS)
# - Each worker binds to ONE GPU and repeatedly pops a job from queue
# - Queue + result TSV live on shared filesystem
# - Uses mkdir-based lock (NFS-safe in practice) for queue/result writes
# ================================================================

set -euo pipefail

print_usage() {
    echo "用法: bash cluster_worker.sh --repo-dir DIR --host-label LABEL --gpu ID --python-bin PY \\" 
    echo "                      --queue-file FILE --queue-lock-dir DIR \\" 
    echo "                      --result-tsv FILE --result-lock-dir DIR --log-dir DIR"
}

REPO_DIR=""
HOST_LABEL=""
GPU_ID=""
PYTHON_BIN=""
QUEUE_FILE=""
QUEUE_LOCK_DIR=""
RESULT_TSV=""
RESULT_LOCK_DIR=""
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir) REPO_DIR="$2"; shift 2;;
        --host-label) HOST_LABEL="$2"; shift 2;;
        --gpu) GPU_ID="$2"; shift 2;;
        --python-bin) PYTHON_BIN="$2"; shift 2;;
        --queue-file) QUEUE_FILE="$2"; shift 2;;
        --queue-lock-dir) QUEUE_LOCK_DIR="$2"; shift 2;;
        --result-tsv) RESULT_TSV="$2"; shift 2;;
        --result-lock-dir) RESULT_LOCK_DIR="$2"; shift 2;;
        --log-dir) LOG_DIR="$2"; shift 2;;
        -h|--help) print_usage; exit 0;;
        *) echo "未知参数: $1"; print_usage; exit 1;;
    esac
done

if [[ -z "$REPO_DIR" || -z "$HOST_LABEL" || -z "$GPU_ID" || -z "$PYTHON_BIN" || -z "$QUEUE_FILE" || -z "$QUEUE_LOCK_DIR" || -z "$RESULT_TSV" || -z "$RESULT_LOCK_DIR" || -z "$LOG_DIR" ]]; then
    echo "参数不完整"
    print_usage
    exit 1
fi

mkdir -p "$LOG_DIR"

pop_job() {
    local job=""
    while ! mkdir "$QUEUE_LOCK_DIR" 2>/dev/null; do
        sleep 0.2
    done

    if [[ -f "$QUEUE_FILE" ]]; then
        job=$(head -n 1 "$QUEUE_FILE" || true)
        if [[ -n "$job" ]]; then
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" 2>/dev/null || true
            mv "$QUEUE_FILE.tmp" "$QUEUE_FILE" 2>/dev/null || true
        fi
    fi

    rmdir "$QUEUE_LOCK_DIR" 2>/dev/null || true
    echo "$job"
}

append_result() {
    local line="$1"
    while ! mkdir "$RESULT_LOCK_DIR" 2>/dev/null; do
        sleep 0.2
    done
    echo -e "$line" >> "$RESULT_TSV"
    rmdir "$RESULT_LOCK_DIR" 2>/dev/null || true
}

# scenario -> (frame_stride, n_iter)
get_params() {
    local scenario="$1"
    case "$scenario" in
        unitree_z1_stackbox) echo "4 12";;
        unitree_z1_dual_arm_stackbox) echo "4 7";;
        unitree_z1_dual_arm_stackbox_v2) echo "4 11";;
        unitree_z1_dual_arm_cleanup_pencils) echo "4 8";;
        unitree_g1_pack_camera) echo "6 11";;
        *) echo "4 11";;
    esac
}

run_one() {
    local scenario="$1"
    local case_id="$2"
    local case_name="${scenario}-${case_id}"

    local case_dir="$REPO_DIR/$scenario/$case_id"
    local ckpt_path="$REPO_DIR/ckpts/unifolm_wma_dual.ckpt"
    local config_path="$REPO_DIR/configs/inference/world_model_interaction.yaml"
    local save_dir="$case_dir/output"
    local prompt_dir="$case_dir/world_model_interaction_prompts"

    local params
    params=$(get_params "$scenario")
    local frame_stride
    local n_iter
    frame_stride=$(echo "$params" | awk '{print $1}')
    n_iter=$(echo "$params" | awk '{print $2}')

    local log_file="$LOG_DIR/${HOST_LABEL}_gpu${GPU_ID}_${case_name}.log"

    local start
    start=$(date +%s)

    local exit_code=0
    if [[ ! -d "$case_dir" ]]; then
        echo "缺少 case_dir: $case_dir" > "$log_file"
        exit_code=10
    elif [[ ! -f "$ckpt_path" ]]; then
        echo "缺少 ckpt: $ckpt_path" > "$log_file"
        exit_code=11
    elif [[ ! -f "$config_path" ]]; then
        echo "缺少 config: $config_path" > "$log_file"
        exit_code=12
    elif [[ ! -d "$prompt_dir" ]]; then
        echo "缺少 prompt_dir: $prompt_dir" > "$log_file"
        exit_code=13
    else
        cd "$REPO_DIR"
        set +e
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluation/world_model_interaction.py \
            --seed 123 \
            --ckpt_path "$ckpt_path" \
            --config "$config_path" \
            --savedir "$save_dir" \
            --bs 1 --height 320 --width 512 \
            --unconditional_guidance_scale 1.0 \
            --ddim_steps 50 \
            --ddim_eta 1.0 \
            --prompt_dir "$prompt_dir" \
            --dataset "$scenario" \
            --video_length 16 \
            --frame_stride "$frame_stride" \
            --n_action_steps 16 \
            --exe_steps 16 \
            --n_iter "$n_iter" \
            --timestep_spacing 'uniform_trailing' \
            --guidance_rescale 0.7 \
            --perframe_ae \
            > "$log_file" 2>&1
        exit_code=$?
        set -e
    fi

    local end
    end=$(date +%s)
    local elapsed=$((end - start))
    local time_desc="$((elapsed / 60))m $((elapsed % 60))s"

    local psnr_val="N/A"
    if [[ $exit_code -ne 0 ]]; then
        psnr_val="ERR(exit=$exit_code)"
    else
        local pred=""
        shopt -s nullglob
        local preds=("$case_dir"/output/inference/*.mp4)
        shopt -u nullglob
        if [[ ${#preds[@]} -gt 0 ]]; then
            pred="${preds[0]}"
        fi
        local gt="$case_dir/${scenario}_${case_id}.mp4"
        local score_json="$case_dir/psnr_score.json"
        if [[ -f "$pred" && -f "$gt" ]]; then
            "$PYTHON_BIN" "$REPO_DIR/psnr_score_for_challenge.py" --gt_video="$gt" --pred_video="$pred" --output_file="$score_json" > /dev/null 2>&1 || true
            psnr_val=$($PYTHON_BIN -c "import json; print(json.load(open('$score_json'))['psnr'])" 2>/dev/null || echo "Err")
        fi
    fi

    # TSV: host\tgpu\tcase\ttime\tpsnr\texit_code
    append_result "${HOST_LABEL}\t${GPU_ID}\t${case_name}\t${time_desc}\t${psnr_val}\t${exit_code}"
}

# announce
append_result "${HOST_LABEL}\t${GPU_ID}\t__worker_start__\t$(date)\tN/A\t0"

while true; do
    job=$(pop_job)
    if [[ -z "$job" ]]; then
        break
    fi

    scenario=$(echo "$job" | cut -d':' -f1)
    case_id=$(echo "$job" | cut -d':' -f2)

    run_one "$scenario" "$case_id"

done

append_result "${HOST_LABEL}\t${GPU_ID}\t__worker_done__\t$(date)\tN/A\t0"

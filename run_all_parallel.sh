#!/bin/bash

# =================================================================
# ASC26 双卡并行基准测试脚本 (Multi-GPU Parallel Baseline)
# 用于将 20 个 Case 分配到 2 张 GPU 上同时运行
# =================================================================

SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

# 获取所有 Case 的列表
ALL_CASES=()
for s in "${SCENARIOS[@]}"; do
    for c in "${CASES[@]}"; do
        ALL_CASES+=("$s:$c")
    done
done

ROOT_DIR=$(pwd)
REPORT_FILE="$ROOT_DIR/benchmark_report.log"

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
    local RUN_SH="$CASE_DIR/run_world_model_interaction.sh"

    echo "[GPU $gpu_id] 正在启动: $CASE_NAME"

    # 执行推理 (强制指定 CUDA_VISIBLE_DEVICES)
    START_TIME=$(date +%s)
    
    cd "$ROOT_DIR"
    # 这里通过环境变量强制脚本在指定卡上运行
    CUDA_VISIBLE_DEVICES=$gpu_id bash "$RUN_SH" > "$CASE_DIR/parallel_run.log" 2>&1
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    TIME_DESC="$(($ELAPSED / 60))m $(($ELAPSED % 60))s"

    # 计算 PSNR
    PRED_VIDEO=$(ls "$CASE_DIR"/output/inference/*.mp4 2>/dev/null | head -n 1)
    GT_VIDEO="$CASE_DIR/${scenario}_${case_id}.mp4"
    SCORE_JSON="$CASE_DIR/psnr_score.json"
    
    PSNR_VAL="N/A"
    if [ -f "$PRED_VIDEO" ] && [ -f "$GT_VIDEO" ]; then
        python3 psnr_score_for_challenge.py --gt_video="$GT_VIDEO" --pred_video="$PRED_VIDEO" --output_file="$SCORE_JSON" > /dev/null 2>&1
        PSNR_VAL=$(python3 -c "import json; print(json.load(open('$SCORE_JSON'))['psnr'])" 2>/dev/null || echo "Err")
    fi

    # 写入临时结果文件（避免并发写入冲突）
    echo "$CASE_NAME | $TIME_DESC | $PSNR_VAL | GPU $gpu_id" >> "${REPORT_FILE}.tmp"
    echo "[GPU $gpu_id] 完成: $CASE_NAME (耗时: $TIME_DESC)"
}

# 记录总起始时间
TOTAL_START=$(date +%s)

# 分配原则：
# GPU 0 控制偶数索引数据，GPU 1 控制奇数索引数据
echo ">>> 开始双卡并行推理 (GPU 0 & GPU 1)..."

for i in "${!ALL_CASES[@]}"; do
    gpu=$((i % 2))
    run_case_on_gpu $gpu "${ALL_CASES[$i]}" &
    
    # 因为推理很慢，所以不需要延时。但为了日志可读性，每启动一卡稍微顿一下。
    sleep 2
done

# 等待所有后台进程结束
wait

# 汇总结果
echo "场景-Case                                | 耗时           | PSNR     | 运行设备" >> "$REPORT_FILE"
echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
sort "${REPORT_FILE}.tmp" >> "$REPORT_FILE"
rm "${REPORT_FILE}.tmp"

# 计算总耗时
TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_DESC="$(($TOTAL_DIFF / 3600))h $((($TOTAL_DIFF % 3600) / 60))m $(($TOTAL_DIFF % 60))s"

echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
echo "双卡总并发执行效率时长: $TOTAL_DESC" >> "$REPORT_FILE"
echo "完成时间: $(date)" >> "$REPORT_FILE"

echo -e "\n[!!!] 双卡并行运行完毕！"
echo "[!!!] 请查看报告: $REPORT_FILE"

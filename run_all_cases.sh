#!/bin/bash

# =================================================================
# ASC26 基准测试脚本 (Baseline Benchmark Script)
# 根据 manual.md 要求编写，用于跑完 5 个场景共 20 个 case
# =================================================================

# --- 1. 定义场景和 Case ---
SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

ROOT_DIR=$(pwd)
REPORT_FILE="$ROOT_DIR/benchmark_report.log"

# 初始化报告文件
echo "================ ASC26 性能基准测试报告 (Baseline) ================" > "$REPORT_FILE"
echo "开始时间: $(date)" >> "$REPORT_FILE"
echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
printf "%-40s | %-15s | %-8s\n" "场景-Case" "耗时 (Real Time)" "PSNR" >> "$REPORT_FILE"
echo "-------------------------------------------------------------------" >> "$REPORT_FILE"

# 记录总起始时间
TOTAL_START=$(date +%s)

# --- 2. 循环处理每个场景 ---
for scenario in "${SCENARIOS[@]}"; do
    echo ">>> 正在进入场景: $scenario"
    
    for case_id in "${CASES[@]}"; do
        CASE_NAME="${scenario}-${case_id}"
        CASE_DIR="$ROOT_DIR/$scenario/$case_id"
        RUN_SH="$CASE_DIR/run_world_model_interaction.sh"
        
        echo "[*] 处理中: $CASE_NAME"

        if [ ! -f "$RUN_SH" ]; then
            echo "[!] 警告: 找不到脚本 $RUN_SH，跳过..."
            printf "%-40s | %-15s | %-8s\n" "$CASE_NAME" "Skipped" "N/A" >> "$REPORT_FILE"
            continue
        fi

        # --- 执行推理 ---
        # 手册要求：bash {scenario_name}/{case_id}/run_world_model_interaction.sh
        cd "$ROOT_DIR"
        
        # 记录单个 case 时间
        START_TIME=$(date +%s)
        
        echo "    [>] 运行推理脚本..."
        bash "$RUN_SH"
        
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        TIME_DESC="$(($ELAPSED / 60))m $(($ELAPSED % 60))s"

        # --- 计算 PSNR ---
        echo "    [>] 计算 PSNR 分数..."
        # 寻找生成的视频（通常在 output/inference/ 目录下）
        # 注意：这里假设输出视频由脚本默认生成
        PRED_VIDEO=$(ls "$CASE_DIR"/output/inference/*.mp4 2>/dev/null | head -n 1)
        # GT视频路径通常在场景文件夹下
        GT_VIDEO="$CASE_DIR/${scenario}_${case_id}.mp4"
        SCORE_JSON="$CASE_DIR/psnr_score.json"
        
        PSNR_VAL="N/A"
        if [ -f "$PRED_VIDEO" ] && [ -f "$GT_VIDEO" ]; then
            python3 psnr_score_for_challenge.py \
                --gt_video="$GT_VIDEO" \
                --pred_video="$PRED_VIDEO" \
                --output_file="$SCORE_JSON"
            
            # 提取 JSON 中的 PSNR 值
            PSNR_VAL=$(python3 -c "import json; print(record['psnr']) if (record := json.load(open('$SCORE_JSON'))) else print('Err')" 2>/dev/null || echo "Err")
        else
            echo "    [!] 错误: 找不到预测视频或参考视频，无法计算 PSNR。"
        fi

        # --- 记录结果 ---
        printf "%-40s | %-15s | %-8s\n" "$CASE_NAME" "$TIME_DESC" "$PSNR_VAL" >> "$REPORT_FILE"
        echo "[+] $CASE_NAME 完成！耗时: $TIME_DESC, PSNR: $PSNR_VAL"
        echo "-------------------------------------------------------------------"
    done
done

# --- 3. 统计总耗时 ---
TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_DESC="$(($TOTAL_DIFF / 3600))h $((($TOTAL_DIFF % 3600) / 60))m $(($TOTAL_DIFF % 60))s"

echo "-------------------------------------------------------------------" >> "$REPORT_FILE"
echo "总基准测试耗时: $TOTAL_DESC" >> "$REPORT_FILE"
echo "完成时间: $(date)" >> "$REPORT_FILE"
echo "===================================================================" >> "$REPORT_FILE"

echo -e "\n[!!!] 所有 Case 运行完毕！"
echo "[!!!] 基准测试报告已生成: $REPORT_FILE"

#!/bin/bash

# --- 1. 定义场景和 Case ---
SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

ROOT_DIR=$(pwd)
BASE_CONFIG="$ROOT_DIR/configs/inference/world_model_interaction.yaml"
REPORT_FILE="$ROOT_DIR/benchmark_report.log"

# 初始化报告文件（带表格格式）
echo "================ ASC26 性能基准测试报告 ================" > "$REPORT_FILE"
echo "开始时间: $(date)" >> "$REPORT_FILE"
echo "-------------------------------------------------------" >> "$REPORT_FILE"
printf "%-40s | %-12s | %-8s\n" "场景-Case" "耗时" "PSNR" >> "$REPORT_FILE"
echo "-------------------------------------------------------" >> "$REPORT_FILE"

# 记录总起始时间
TOTAL_START=$(date +%s)

# --- 2. 循环处理 ---
for scenario in "${SCENARIOS[@]}"; do
    for case_id in "${CASES[@]}"; do
        CASE_NAME="${scenario}-${case_id}"
        echo "[*] 正在处理: $CASE_NAME"
        
        CASE_PATH="$ROOT_DIR/$scenario/$case_id"
        ORIGINAL_SH="$CASE_PATH/run_world_model_interaction.sh"
        TEMP_YAML="$CASE_PATH/temp.yaml"
        TEMP_SH="$CASE_PATH/temp_run.sh"

        if [ ! -f "$ORIGINAL_SH" ]; then
            echo "[!] 跳过: 找不到 $ORIGINAL_SH"
            printf "%-40s | %-12s | %-8s\n" "$CASE_NAME" "跳过" "N/A" >> "$REPORT_FILE"
            continue
        fi

        # --- 3. 生成 temp.yaml ---
        python3 - <<EOF
import yaml
with open('$BASE_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['data']['params']['test']['params']['data_dir'] = '$CASE_PATH/world_model_interaction_prompts'
config['data']['params']['dataset_and_weights'] = { '$scenario': 1.0 }
with open('$TEMP_YAML', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
EOF

        # --- 4. 生成并修补 temp_run.sh ---
        cp "$ORIGINAL_SH" "$TEMP_SH"
        sed -i "s|res_dir=.*|res_dir=\"$scenario/$case_id\"|g" "$TEMP_SH"
        sed -i "s|dataset=.*|dataset=\"$scenario\"|g" "$TEMP_SH"   # 保留 dataset 替换（防止原始脚本依赖此变量）
        sed -i "s|--config .*|--config \"$TEMP_YAML\" \\\\|g" "$TEMP_SH"
        sed -i "s|--prompt_dir .*|--prompt_dir \"$scenario/$case_id/world_model_interaction_prompts\" \\\\|g" "$TEMP_SH"

        # --- 5. 推理计时开始 ---
        CASE_START=$(date +%s)

        echo "[>] 执行推理: bash $scenario/$case_id/temp_run.sh"
        bash "$TEMP_SH"   # 不静默输出，完整显示推理日志，便于调试和观察

        # --- 6. 计算 PSNR ---
        echo "[>] 计算 PSNR 得分..."
        PRED_VIDEO=$(ls $CASE_PATH/output/inference/*.mp4 2>/dev/null | head -n 1)
        GT_VIDEO="$CASE_PATH/${scenario}_${case_id}.mp4"
        SCORE_JSON="$CASE_PATH/psnr_score.json"
        
        PSNR_VAL="N/A"
        if [ -f "$PRED_VIDEO" ] && [ -f "$GT_VIDEO" ]; then
            python -B psnr_score_for_challenge.py \
                --gt_video="$GT_VIDEO" \
                --pred_video="$PRED_VIDEO" \
                --output_file="$SCORE_JSON"

            # 使用 Python 解析 JSON（跨平台，macOS/Linux 均兼容）
            PSNR_VAL=$(python3 -c "import json; print(json.load(open('$SCORE_JSON'))['psnr'])" 2>/dev/null || echo "Err")
        else
            echo "[!] 警告: 缺少预测视频或 GT 视频，无法计算 PSNR"
            if [ ! -f "$PRED_VIDEO" ]; then echo "    - 预测视频未找到: $CASE_PATH/output/inference/*.mp4"; fi
            if [ ! -f "$GT_VIDEO" ]; then echo "    - GT 视频未找到: $GT_VIDEO"; fi
            PSNR_VAL="Missing"
        fi

        # --- 7. 推理计时结束并记录 ---
        CASE_END=$(date +%s)
        CASE_DIFF=$((CASE_END - CASE_START))
        CASE_TIME="$(($CASE_DIFF / 60))分$(($CASE_DIFF % 60))秒"

        printf "%-40s | %-12s | %-8s\n" "$CASE_NAME" "$CASE_TIME" "$PSNR_VAL" >> "$REPORT_FILE"
        echo "[+] $CASE_NAME 完成，耗时: $CASE_TIME，PSNR: $PSNR_VAL"
        echo "-------------------------------------------------------"
    done
done

# --- 8. 统计总耗时 ---
TOTAL_END=$(date +%s)
TOTAL_DIFF=$((TOTAL_END - TOTAL_START))
TOTAL_TIME="$(($TOTAL_DIFF / 3600))小时$((($TOTAL_DIFF % 3600) / 60))分$(($TOTAL_DIFF % 60))秒"

echo "-------------------------------------------------------" >> "$REPORT_FILE"
echo "测试总耗时: $TOTAL_TIME" >> "$REPORT_FILE"
echo "结束时间: $(date)" >> "$REPORT_FILE"

echo "[!!!] 所有的 Case 运行完毕！"
echo "[!!!] 请查看报表文件: $REPORT_FILE"
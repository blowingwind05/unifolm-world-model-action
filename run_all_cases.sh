#!/bin/bash

# --- 1. 定义场景和 Case ---
SCENARIOS=("unitree_g1_pack_camera" "unitree_z1_stackbox" "unitree_z1_dual_arm_stackbox" "unitree_z1_dual_arm_stackbox_v2" "unitree_z1_dual_arm_cleanup_pencils")
CASES=("case1" "case2" "case3" "case4")

ROOT_DIR=$(pwd)
BASE_CONFIG="$ROOT_DIR/configs/inference/world_model_interaction.yaml"
LOG_FILE="$ROOT_DIR/benchmark_baseline.log"

echo "ASC26 Benchmark Started at $(date)" > "$LOG_FILE"
echo "--------------------------------------" >> "$LOG_FILE"

# --- 2. 循环处理 ---
for scenario in "${SCENARIOS[@]}"; do
    for case_id in "${CASES[@]}"; do
        echo "[*] Current Target: $scenario / $case_id"
        
        CASE_PATH="$ROOT_DIR/$scenario/$case_id"
        ORIGINAL_SH="$CASE_PATH/run_world_model_interaction.sh"
        TEMP_YAML="$CASE_PATH/temp.yaml"
        TEMP_SH="$CASE_PATH/temp_run.sh"

        # 检查原始 .sh 文件是否存在
        if [ ! -f "$ORIGINAL_SH" ]; then
            echo "[!] Warning: $ORIGINAL_SH not found, skipping."
            continue
        fi

        # --- 3. 生成 temp.yaml ---
        # 修正 data_dir 为当前 case 的路径，并过滤 dataset_and_weights
        python3 - <<EOF
import yaml
import os
with open('$BASE_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# 更新路径和权重
config['data']['params']['test']['params']['data_dir'] = '$CASE_PATH/world_model_interaction_prompts'
config['data']['params']['dataset_and_weights'] = { '$scenario': 1.0 }

with open('$TEMP_YAML', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
EOF

        # --- 4. 生成并修补 temp_run.sh ---
        # 将原始 sh 复制一份，并用 sed 替换关键路径
        cp "$ORIGINAL_SH" "$TEMP_SH"
        
        # 修正脚本内的变量定义和 config 路径
        # 使用 | 作为 sed 分隔符以处理路径中的斜杠
        sed -i "s|res_dir=.*|res_dir=\"$scenario/$case_id\"|g" "$TEMP_SH"
        sed -i "s|dataset=.*|dataset=\"$scenario\"|g" "$TEMP_SH"
        sed -i "s|--config .*|--config \"$TEMP_YAML\" \\\\|g" "$TEMP_SH"
        sed -i "s|--prompt_dir .*|--prompt_dir \"$scenario/$case_id/world_model_interaction_prompts\" \\\\|g" "$TEMP_SH"

        # --- 5. 执行推理 ---
        echo "[>] Executing: bash $scenario/$case_id/temp_run.sh"
        bash "$TEMP_SH"

        # --- 6. 运行 PSNR 评分 ---
        echo "[>] Calculating PSNR Score..."
        # 匹配生成的预测视频 (通常在 output/inference 目录下)
        PRED_VIDEO=$(ls $CASE_PATH/output/inference/*.mp4 2>/dev/null | head -n 1)
        # GT 视频路径，根据你的描述拼接
        GT_VIDEO="$CASE_PATH/${scenario}_${case_id}.mp4"
        SCORE_JSON="$CASE_PATH/psnr_score.json"

        if [ -f "$PRED_VIDEO" ] && [ -f "$GT_VIDEO" ]; then
            python -B psnr_score_for_challenge.py \
                --gt_video="$GT_VIDEO" \
                --pred_video="$PRED_VIDEO" \
                --output_file="$SCORE_JSON"
            
            # 记录结果
            SCORE=$(grep "psnr" "$SCORE_JSON" || echo "Error calculating score")
            echo "$scenario-$case_id: $SCORE" >> "$LOG_FILE"
        else
            echo "$scenario-$case_id: FAILED (Missing video files)" >> "$LOG_FILE"
        fi

        echo "[+] Done $scenario-$case_id"
        echo "--------------------------------------"
    done
done

echo "Benchmark Complete. Results in $LOG_FILE"
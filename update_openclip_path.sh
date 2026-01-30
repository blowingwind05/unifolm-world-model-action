#!/bin/bash

# 获取仓库根目录（脚本所在目录即 repo root）
REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
TARGET_PATH="$REPO_ROOT/openclip/open_clip_pytorch_model.bin"
CONFIG_FILE="configs/inference/world_model_interaction.yaml"

echo "正在修改配置文件: $CONFIG_FILE"
echo "目标模型路径: $TARGET_PATH"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

# 使用 Python 处理 YAML 的智能修改（支持替换现有项或新增缺失项）
python3 -c "
import sys

config_file = '$CONFIG_FILE'
target_path = '$TARGET_PATH'

with open(config_file, 'r') as f:
    lines = f.readlines()

output_lines = []
inside_target = False
inside_params = False
seen_version_in_block = False
param_indent = ''
base_indent_len = 0

# 定义需要修改的目标类名
target_classes = [
    'unifolm_wma.modules.encoders.condition.FrozenOpenCLIPEmbedder',
    'unifolm_wma.modules.encoders.condition.FrozenOpenCLIPImageEmbedderV2'
]

line_idx = 0
while line_idx < len(lines):
    line = lines[line_idx]
    
    # 1. 检查是否进入目标 target 块
    is_target_line = any(f'target: {t}' in line for t in target_classes)
    
    if is_target_line:
        inside_target = True
        inside_params = False
        seen_version_in_block = False
        output_lines.append(line)
        line_idx += 1
        continue
        
    if inside_target:
        # 2. 寻找 params
        if 'params:' in line:
            inside_params = True
            output_lines.append(line)
            
            # 计算缩进
            base_indent = line.split('params:')[0]
            base_indent_len = len(base_indent)
            # 默认子项缩进增加 2 个空格
            param_indent = base_indent + '  '
            
            line_idx += 1
            continue
            
        if inside_params:
            stripped = line.strip()
            # 3. 检查是否跳出了 params 块
            # 如果是非空行 且 缩进小于等于 params 的缩进，说明块结束
            if stripped and not stripped.startswith('#'):
                curr_indent_len = len(line) - len(line.lstrip())
                if curr_indent_len <= base_indent_len:
                    # 块结束了，如果没有 version，插入它
                    if not seen_version_in_block:
                        output_lines.append(f'{param_indent}version: \"{target_path}\"\n')
                    
                    # 重置状态
                    inside_target = False
                    inside_params = False
                    output_lines.append(line)
                    line_idx += 1
                    continue

            # 4. 检查是否是 version 行，如果是则替换
            if stripped.startswith('version:'):
                # 保持原有的缩进（通常就是 param_indent）
                curr_indent = line.split('version:')[0]
                output_lines.append(f'{curr_indent}version: \"{target_path}\"\n')
                seen_version_in_block = True
                line_idx += 1
                continue

    output_lines.append(line)
    line_idx += 1

# 写回文件
with open(config_file, 'w') as f:
    f.writelines(output_lines)
"

echo "✅ 修改完成！"
grep "version:" "$CONFIG_FILE"

# 检查模型文件是否存在，如果不在目标位置但在 ckpts 下，提示移动
if [ ! -f "$TARGET_PATH" ]; then
    echo "⚠️  警告: 目标路径下没有找到模型文件: $TARGET_PATH"
    CKPT_PATH="$REPO_ROOT/ckpts/open_clip_pytorch_model.bin"
    if [ -f "$CKPT_PATH" ]; then
        echo "ℹ️  发现模型文件在 ckpts 目录: $CKPT_PATH"
        echo "   你可以运行以下命令移动它："
        echo "   mkdir -p $(dirname $TARGET_PATH) && mv $CKPT_PATH $TARGET_PATH"
    fi
else
    echo "✅ 模型文件已就位。"
fi

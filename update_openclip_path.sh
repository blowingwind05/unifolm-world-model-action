#!/bin/bash

# 获取当前用户的 HOME 目录
USER_HOME=$HOME
TARGET_PATH="$USER_HOME/unifolm-world-model-action/openclip/open_clip_pytorch_model.bin"
CONFIG_FILE="configs/inference/world_model_interaction.yaml"

echo "正在修改配置文件: $CONFIG_FILE"
echo "目标模型路径: $TARGET_PATH"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

# 使用 sed 替换 version 参数
# 注意：这里会替换文件中所有的 version: "..." 行
# 根据之前的检查，该文件中只有 OpenCLIP 相关的配置使用了 version 参数，所以是安全的
sed -i "s|version: \".*\"|version: \"$TARGET_PATH\"|g" "$CONFIG_FILE"

echo "✅ 修改完成！"
grep "version:" "$CONFIG_FILE"

# 检查模型文件是否存在，如果不在目标位置但在 ckpts 下，提示移动
if [ ! -f "$TARGET_PATH" ]; then
    echo "⚠️  警告: 目标路径下没有找到模型文件: $TARGET_PATH"
    CKPT_PATH="$USER_HOME/unifolm-world-model-action/ckpts/open_clip_pytorch_model.bin"
    if [ -f "$CKPT_PATH" ]; then
        echo "ℹ️  发现模型文件在 ckpts 目录: $CKPT_PATH"
        echo "   你可以运行以下命令移动它："
        echo "   mkdir -p $(dirname $TARGET_PATH) && mv $CKPT_PATH $TARGET_PATH"
    fi
else
    echo "✅ 模型文件已就位。"
fi

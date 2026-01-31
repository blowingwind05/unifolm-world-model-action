#!/bin/bash
set -euo pipefail

# Torch profiler 封装脚本：unitree_g1_pack_camera/case1
# - 单机单卡
# - 只抓一个较小的 profiling 窗口，避免 trace 过大/过慢

usage() {
  cat <<'EOF'
用法：
  bash profile_case1_torchprof.sh [选项]

选项：
  --gpu ID           使用的 GPU（默认：0）
  --python BIN       Python 可执行文件（默认：python3）
  --with-stack       记录 Python 调用栈（更慢、trace 更大，定位更精确）
  --light            轻量采集（更小更快，推荐先用它验证/看热点）
  --clean            运行前清理旧的 trace（避免 TensorBoard 扫描超大目录卡死）
  --steps N          覆盖 n_iter（默认：11）
  --ddim-steps N     覆盖 ddim_steps（默认：50）
  --tb               运行结束后自动启动 TensorBoard
  --tb-port PORT     TensorBoard 端口（默认：6006）
  -h, --help         显示帮助

输出：
  Trace 目录：unitree_g1_pack_camera/case1/output/torch_profiler
  运行日志：unitree_g1_pack_camera/case1/output/torch_profiler/run.log

说明：
  - 本脚本以外层循环 `tqdm(range(n_iter))` 为 step，采集窗口为 wait=1 / warmup=1 / active=2。
  - 真实项目里一次 step 会包含 DDIM 采样（例如 50 steps）两次调用，事件数非常多。
    如果 TensorBoard 一直加载/卡死，优先用 --light 或降低 --ddim-steps/--steps。
  - 如果你只想看端到端 wall time（不做 profiler），请运行：
      bash unitree_g1_pack_camera/case1/run_world_model_interaction.sh
EOF
}

GPU_ID=0
PYTHON_BIN=python3
WITH_STACK=0
LIGHT=0
CLEAN=0
N_ITER=11
DDIM_STEPS=50
LAUNCH_TB=0
TB_PORT=6006

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --with-stack) WITH_STACK=1; shift 1;;
    --light) LIGHT=1; shift 1;;
    --clean) CLEAN=1; shift 1;;
    --steps) N_ITER="$2"; shift 2;;
    --ddim-steps) DDIM_STEPS="$2"; shift 2;;
    --tb) LAUNCH_TB=1; shift 1;;
    --tb-port) TB_PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

CASE_DIR="unitree_g1_pack_camera/case1"
LOGDIR="$CASE_DIR/output/torch_profiler"
mkdir -p "$LOGDIR"

if [[ $CLEAN -eq 1 ]]; then
  echo "[TorchProfiler] 清理旧 trace：$LOGDIR/*.pt.trace.json" | tee -a "$LOGDIR/run.log" || true
  rm -f "$LOGDIR"/*.pt.trace.json 2>/dev/null || true
fi

RUN_LOG="$LOGDIR/run.log"

echo "[TorchProfiler] 仓库根目录: $ROOT_DIR" | tee "$RUN_LOG"
echo "[TorchProfiler] Python: $PYTHON_BIN" | tee -a "$RUN_LOG"
echo "[TorchProfiler] GPU: $GPU_ID" | tee -a "$RUN_LOG"
echo "[TorchProfiler] Trace 输出目录: $LOGDIR" | tee -a "$RUN_LOG"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<PY 2>&1 | tee -a "$RUN_LOG"
import os, sys, runpy
import ctypes
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

logdir = os.path.abspath("$LOGDIR")
os.makedirs(logdir, exist_ok=True)

# 通过 runpy.run_path 执行脚本时，Python 的模块搜索路径不会自动包含 scripts/evaluation。
# 这里显式把仓库根目录 + scripts/evaluation 放到 sys.path 前面，保证能 import eval_utils。
repo_root = os.path.abspath(os.getcwd())
eval_dir = os.path.join(repo_root, "scripts", "evaluation")
if eval_dir not in sys.path:
  sys.path.insert(0, eval_dir)
if repo_root not in sys.path:
  sys.path.insert(0, repo_root)

# 检测 CUPTI：没有 CUPTI 时，PyTorch profiler 的 CUDA(Device) 侧耗时通常会全部显示为 0。
try:
  import torch
  if torch.cuda.is_available():
    try:
      ctypes.CDLL("libcupti.so")
    except OSError:
      print("[TorchProfiler][警告] 未找到 libcupti.so（CUPTI）。")
      print("[TorchProfiler][警告] 这会导致 TensorBoard/Profiler 的 Device self/total duration 全为 0。")
      print("[TorchProfiler][建议] 在当前 conda 环境安装 CUPTI：conda install -c nvidia cupti")
      print("[TorchProfiler][建议] 或 pip（CUDA12 系常见）：pip install nvidia-cuda-cupti-cu12")
      print("[TorchProfiler][建议] 安装后重启进程并重新采集 trace。")
except Exception as e:
  print("[TorchProfiler][提示] 无法检测 torch/CUPTI：", repr(e))

with_stack = bool(int("$WITH_STACK"))
light = bool(int("$LIGHT"))

# 轻量模式：减少事件体积，显著提升 TensorBoard 加载速度
record_shapes = (not light)
profile_memory = (not light)
active_steps = 1 if light else 2

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
  schedule=schedule(wait=1, warmup=1, active=active_steps, repeat=1),
    on_trace_ready=tensorboard_trace_handler(logdir),
  record_shapes=record_shapes,
  profile_memory=profile_memory,
    with_stack=with_stack,
)

with prof:
    # 重要：不要把 tqdm.tqdm 替换成函数，否则 pytorch_lightning 在 import 阶段会继承失败。
    # 这里采用更安全的方式：先导入 pytorch_lightning，再仅 patch tqdm.tqdm.__iter__。
    import pytorch_lightning  # noqa: F401
    import tqdm

    orig_iter = tqdm.tqdm.__iter__

    def patched_iter(self):
        for x in orig_iter(self):
            yield x
            prof.step()

    tqdm.tqdm.__iter__ = patched_iter

    sys.argv = [
        "scripts/evaluation/world_model_interaction.py",
        "--seed", "123",
        "--ckpt_path", "ckpts/unifolm_wma_dual.ckpt",
        "--config", "configs/inference/world_model_interaction.yaml",
        "--savedir", "$CASE_DIR/output",
        "--bs", "1", "--height", "320", "--width", "512",
        "--unconditional_guidance_scale", "1.0",
        "--ddim_steps", str(int("$DDIM_STEPS")),
        "--ddim_eta", "1.0",
        "--prompt_dir", "$CASE_DIR/world_model_interaction_prompts",
        "--dataset", "unitree_g1_pack_camera",
        "--video_length", "16",
        "--frame_stride", "6",
        "--n_action_steps", "16",
        "--exe_steps", "16",
        "--n_iter", str(int("$N_ITER")),
        "--timestep_spacing", "uniform_trailing",
        "--guidance_rescale", "0.7",
        "--perframe_ae",
    ]

    runpy.run_path("scripts/evaluation/world_model_interaction.py", run_name="__main__")

print("Profiler trace 已保存到:", logdir)
PY

echo "[TorchProfiler] 完成。Trace 目录: $LOGDIR" | tee -a "$RUN_LOG"

if [[ $LAUNCH_TB -eq 1 ]]; then
  echo "[TorchProfiler] 正在启动 TensorBoard，端口: $TB_PORT" | tee -a "$RUN_LOG"
  # 用同一个 Python 环境启动 TensorBoard，避免系统 tensorboard 与 conda 环境不一致。
  if ! "$PYTHON_BIN" -c "import tensorboard" >/dev/null 2>&1; then
    echo "未在当前 Python 环境中找到 tensorboard。可尝试安装：$PYTHON_BIN -m pip install tensorboard" | tee -a "$RUN_LOG"
    exit 3
  fi

  # PyTorch Profiler 的 TensorBoard 展示需要插件 torch-tb-profiler；缺少时通常会显示“ No dashboards are active... ”
  if ! "$PYTHON_BIN" -c "import torch_tb_profiler" >/dev/null 2>&1; then
    echo "未检测到 torch-tb-profiler（TensorBoard 的 PyTorch Profiler 插件）。" | tee -a "$RUN_LOG"
    echo "请先安装：$PYTHON_BIN -m pip install torch-tb-profiler" | tee -a "$RUN_LOG"
    echo "安装后重新运行本脚本的 --tb，或手动启动：$PYTHON_BIN -m tensorboard --logdir $LOGDIR" | tee -a "$RUN_LOG"
    exit 4
  fi

  # 某些环境下 `python -m tensorboard` 可能没有 __main__，这里优先使用 `tensorboard.main`。
  if "$PYTHON_BIN" -c "import tensorboard.main" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m tensorboard.main --logdir "$LOGDIR" --port "$TB_PORT" --bind_all
  elif command -v tensorboard >/dev/null 2>&1; then
    # 回退：使用 PATH 里的 tensorboard 可执行文件
    tensorboard --logdir "$LOGDIR" --port "$TB_PORT" --bind_all
  else
    echo "无法启动 TensorBoard：当前环境缺少可执行入口。建议执行：$PYTHON_BIN -m pip install -U tensorboard" | tee -a "$RUN_LOG"
    exit 5
  fi
fi

#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="python"
CSV_PATH="./data_template.csv"
OUTPUT_DIR="./results"
ENCODING="utf-8"
DEVICE="cpu"
SEED="42"
USER_COL=""
TEXT_COL=""
LABEL_COL=""
CRISIS_COL=""
HISTORY_COL=""
HISTORY_SEP="\n"
LAYER2_THRESHOLD_FACTOR="1.0"
FIXED_THRESHOLD=""
SKIP_INSTALL="0"

print_usage() {
  cat <<'EOF'
用法:
  bash run_all_experiments.sh [选项]

选项:
  --python <path>                     Python 可执行文件（默认: python）
  --csv <path>                        CSV 路径（默认: ./data_template.csv）
  --output-dir <path>                 输出目录（默认: ./results）
  --encoding <enc>                    CSV 编码（默认: utf-8）
  --device <cpu|cuda>                 推理设备（默认: cpu）
  --seed <int>                        随机种子（默认: 42）
  --user-col <name>                   用户列名
  --text-col <name>                   文本列名
  --label-col <name>                  标签列名（0-7）
  --crisis-col <name>                 危机二值列名（0/1）
  --history-col <name>                历史列名
  --history-sep <sep>                 历史分隔符（默认: \n）
  --layer2-threshold-factor <float>   二层阈值缩放（默认: 1.0）
  --fixed-threshold <float>           固定危机阈值（不传则用验证集 best-F1）
  --skip-install                      跳过依赖安装
  -h, --help                          显示帮助
EOF
}

step() {
  local title="$1"
  shift
  echo ""
  echo "========================================================================"
  echo "[步骤] ${title}"
  echo "命令: $*"
  echo "========================================================================"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_EXE="$2"; shift 2 ;;
    --csv) CSV_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --encoding) ENCODING="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --user-col) USER_COL="$2"; shift 2 ;;
    --text-col) TEXT_COL="$2"; shift 2 ;;
    --label-col) LABEL_COL="$2"; shift 2 ;;
    --crisis-col) CRISIS_COL="$2"; shift 2 ;;
    --history-col) HISTORY_COL="$2"; shift 2 ;;
    --history-sep) HISTORY_SEP="$2"; shift 2 ;;
    --layer2-threshold-factor) LAYER2_THRESHOLD_FACTOR="$2"; shift 2 ;;
    --fixed-threshold) FIXED_THRESHOLD="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL="1"; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "未知参数: $1"; print_usage; exit 1 ;;
  esac
done

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "错误: --device 只能是 cpu 或 cuda"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "当前目录: $SCRIPT_DIR"
echo "Python: $PYTHON_EXE"
echo "设备: $DEVICE, 随机种子: $SEED"
echo "CSV: $CSV_PATH"

if [[ "$SKIP_INSTALL" == "0" ]]; then
  step "安装基础依赖（tests/requirements.txt）" \
    "$PYTHON_EXE" -m pip install -r tests/requirements.txt
  step "安装深度学习依赖（torch + transformers）" \
    "$PYTHON_EXE" -m pip install torch transformers
else
  echo "已跳过依赖安装（--skip-install）"
fi

mkdir -p "$OUTPUT_DIR"
REAL_EVAL_JSON="${OUTPUT_DIR}/real_eval_metrics.json"
PROTOCOL_LOG="${OUTPUT_DIR}/protocol_demo.log"
ABLATION_LOG="${OUTPUT_DIR}/ablation_suite.log"
BASELINE_JSON="${OUTPUT_DIR}/baseline_comparison.json"
PAPER_SUMMARY_PREFIX="${OUTPUT_DIR}/paper_result_summary"

step "实验 1/4：协议演示（run_protocol_demo.py）" \
  bash -lc "$PYTHON_EXE experiments/run_protocol_demo.py | tee \"$PROTOCOL_LOG\""

if [[ ! -f "$CSV_PATH" ]]; then
  echo "错误: 未找到 CSV 文件: $CSV_PATH"
  echo "请先编辑 data_template.csv 或传入 --csv。"
  exit 1
fi

EVAL_CMD=(
  "$PYTHON_EXE" "experiments/run_real_dataset_eval.py"
  "--csv" "$CSV_PATH"
  "--encoding" "$ENCODING"
  "--device" "$DEVICE"
  "--seed" "$SEED"
  "--output-json" "$REAL_EVAL_JSON"
  "--history-sep" "$HISTORY_SEP"
  "--layer2-threshold-factor" "$LAYER2_THRESHOLD_FACTOR"
)

if [[ -n "$USER_COL" ]]; then EVAL_CMD+=("--user-col" "$USER_COL"); fi
if [[ -n "$TEXT_COL" ]]; then EVAL_CMD+=("--text-col" "$TEXT_COL"); fi
if [[ -n "$LABEL_COL" ]]; then EVAL_CMD+=("--label-col" "$LABEL_COL"); fi
if [[ -n "$CRISIS_COL" ]]; then EVAL_CMD+=("--crisis-col" "$CRISIS_COL"); fi
if [[ -n "$HISTORY_COL" ]]; then EVAL_CMD+=("--history-col" "$HISTORY_COL"); fi
if [[ -n "$FIXED_THRESHOLD" ]]; then EVAL_CMD+=("--fixed-threshold" "$FIXED_THRESHOLD"); fi

step "实验 2/4：真实数据评估（run_real_dataset_eval.py）" "${EVAL_CMD[@]}"

BASELINE_CMD=(
  "$PYTHON_EXE" "experiments/run_baseline_comparison.py"
  "--csv" "$CSV_PATH"
  "--encoding" "$ENCODING"
  "--device" "$DEVICE"
  "--seed" "$SEED"
  "--history-sep" "$HISTORY_SEP"
  "--layer2-threshold-factor" "$LAYER2_THRESHOLD_FACTOR"
  "--output-json" "$BASELINE_JSON"
)
if [[ -n "$USER_COL" ]]; then BASELINE_CMD+=("--user-col" "$USER_COL"); fi
if [[ -n "$TEXT_COL" ]]; then BASELINE_CMD+=("--text-col" "$TEXT_COL"); fi
if [[ -n "$LABEL_COL" ]]; then BASELINE_CMD+=("--label-col" "$LABEL_COL"); fi
if [[ -n "$CRISIS_COL" ]]; then BASELINE_CMD+=("--crisis-col" "$CRISIS_COL"); fi
if [[ -n "$HISTORY_COL" ]]; then BASELINE_CMD+=("--history-col" "$HISTORY_COL"); fi

step "实验 3/4：基线对比（run_baseline_comparison.py）" "${BASELINE_CMD[@]}"

step "实验 4/4：消融实验（run_ablation_suite.py）" \
  bash -lc "$PYTHON_EXE experiments/run_ablation_suite.py | tee \"$ABLATION_LOG\""

step "导出论文摘要（export_paper_summary.py）" \
  "$PYTHON_EXE" experiments/export_paper_summary.py \
  --input-json "$REAL_EVAL_JSON" \
  --output-prefix "$PAPER_SUMMARY_PREFIX"

echo ""
echo "全部实验已完成。"
echo "真实评估指标: $REAL_EVAL_JSON"
echo "基线对比指标: $BASELINE_JSON"
echo "协议演示日志: $PROTOCOL_LOG"
echo "消融实验日志: $ABLATION_LOG"
echo "论文摘要(中): ${PAPER_SUMMARY_PREFIX}.zh.txt"
echo "论文摘要(英): ${PAPER_SUMMARY_PREFIX}.en.txt"
echo "论文摘要(MD): ${PAPER_SUMMARY_PREFIX}.md"
echo ""
echo "列名不一致时可传映射参数（--user-col/--text-col/--label-col 等）。"
echo "示例:"
echo "bash run_all_experiments.sh --csv ./your.csv --user-col uid --text-col content --label-col emotion_id --crisis-col is_crisis --history-col context --skip-install"

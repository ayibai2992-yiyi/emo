# 学生增强数据集说明（云端同步版）

面向：**学生 ↔ AI/咨询对话** + **学生日常热语**。  
主训练文件：`upload_ready_student_enriched.csv`（约 2 万条，与主实验字段对齐）。

---

## 云端路径约定

同步后建议放在：

```text
/root/emo/data/external_processed/
  README.md                              ← 本说明
  SOURCES.md                             ← 来源与许可/限制
  upload_ready_student_enriched.csv      ← 【主用】混合 20k
  slang_hot_speech.csv                   ← 热语子集
  campus_local_client_turns.csv          ← 校园主题本地筛选子集
  ingest_meta.json                       ← 采集统计
```

原始 CPCD JSON（`data/external_raw/cpcd/`）默认**不同步**（体积大且训练不需要）；需要可本机重跑采集脚本。

---

## 字段说明（与 `upload_ready_20k` 一致）

| 字段 | 含义 |
|------|------|
| `user_id` | 用户 ID（按用户划分 train/val/test） |
| `user_id_type` | `cpcd_student` / `student_slang` / `campus_local` / `base` 等 |
| `text` | 当前轮文本 |
| `label_id` | 情绪 0–7（中性…绝望） |
| `is_crisis` | 危机 0/1 |
| `source_file` | 来源溯源 |
| `label_raw` | 弱标/梗名等备注 |
| `score` | 强度/压力相关分数 |
| `row_id` | 行唯一键 |
| `conversation_history` | 历史轮（文本） |
| `role` | `client`（学生）或 `counselor`（AI/咨询师） |

---

## 混合配比（约）

| 来源 | 约占比 | 说明 |
|------|--------|------|
| CPCD 2026 校园长程咨询 | ~50% | 学生–Counselor 对话，只保留/偏重学生轮 |
| 本地校园主题咨询筛选 | ~25% | 挂科/宿舍/考研等关键词 |
| 原 upload_ready client | ~15% | 保留部分金标风格样本 |
| 热语口语句 | ~9% | 梗转写 + 校园热词固定句 |

当前统计量级（以本机 `ingest_meta.json` 为准）：enriched≈20000，client≈97%，危机率约 2%～3%。

---

## 重要限制（写论文必读）

1. **CPCD 无逐句人工情绪金标**：`label_id` / `is_crisis` 多为 domain + stress + 词典 **弱标注**。  
   - 适合：学生场景适配、`--specialty` 对照实验。  
   - 主表建议：与原 `upload_ready_20k_stratified_users.csv`（有更稳标签）对照，或抽检弱标。
2. **热语**由梗指南模板生成 + 少量固定校园口语，非大规模人工标注贴吧语料。
3. 本地 5.5 万段咨询与 SmileChat 高度同源，故未重复全量下载 Smile。

---

## 云端拉取 / 同步

### 方式 A：Git（推荐）

本机已推到分支 `cloud-minimal` 后，在 AutoDL：

```bash
cd /root/emo
git fetch origin cloud-minimal
git checkout cloud-minimal
git pull origin cloud-minimal
ls -lh data/external_processed/
```

若当前已在 `cloud-minimal` 且有本地改动，用：

```bash
cd /root/emo
git pull --rebase origin cloud-minimal
```

### 方式 B：只拉数据目录（若已 clone）

确认存在：

```bash
ls /root/emo/data/external_processed/upload_ready_student_enriched.csv
```

---

## 推荐训练命令（勿覆盖基线权重）

在 `/root/emo/oiu`：

```bash
python experiments/train_unified_model.py \
  --csv ../data/external_processed/upload_ready_student_enriched.csv \
  --epochs 3 --batch-size 8 --device cuda \
  --freeze-bert-layers 10 --specialty \
  --output-dir models/specialty_enriched \
  --result-dir results
```

评测：

```bash
python experiments/run_real_dataset_eval.py \
  --csv ../data/external_processed/upload_ready_student_enriched.csv \
  --device cuda --force-deep --client-only-metrics \
  --output-json results/real_eval_specialty_enriched.json
```

与旧基线对照时，旧数据仍用：

```text
../data/upload_ready_20k_stratified_users.csv
```

---

## 本机重新生成

```bash
# Windows PowerShell
$env:HF_ENDPOINT='https://hf-mirror.com'
python oiu/scripts/ingest_external_student_datasets.py --cpcd-students 10 --max-enriched 20000 --client-only-cpcd
```

详见 `SOURCES.md`。

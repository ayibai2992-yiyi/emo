# 外部学生数据来源说明

## 已落地文件

| 文件 | 约大小 | 说明 |
|------|--------|------|
| `upload_ready_student_enriched.csv` | ~29 MB | **主用**混合 20k，云端训练入口 |
| `cpcd_student_ai_turns.csv` | ~97 MB | CPCD 全量学生轮（可选，云端默认可不同步） |
| `slang_hot_speech.csv` | ~0.3 MB | 学生口吻热语短句 |
| `campus_local_client_turns.csv` | ~6 MB | 本地咨询校园主题筛选 |
| `ingest_meta.json` | 小 | 采集统计 |
| `README.md` | — | 云端使用说明（配比/字段/命令） |

原始 JSON：`data/external_raw/cpcd/`（约 45 MB，训练不需要，默认不入库）。

## 来源

1. **CPCD / Psy-Chronicle（2026，推荐）**  
   - HuggingFace: https://huggingface.co/datasets/EdwinUstb/CPCD  
   - 论文: https://arxiv.org/abs/2605.22140  
   - 校园长程心理咨询，约 100 名学生画像 + 学期轨迹  
   - 本仓库经 `hf-mirror.com` 下载 `conversation/1..10`（每桶约 10 人）

2. **本地梗指南 → 热语**  
   - `数据/data/raw/梗数据集（思考怎么处理中）.md`  
   - 转写为「学生怎么说话」短句，而非百科解释  

3. **本地时序咨询校园筛选**  
   - `数据/data/light_deal/train/时序性对话`（规模接近 SmileChat 5.5 万段）  
   - 按挂科/宿舍/考研等关键词筛校园主题  

## 重要限制

- CPCD **无逐句人工情绪金标**；当前 `label_id`/`is_crisis` 为 domain+stress+词典 **弱标注**。  
  论文主表请抽检或与原 20k 有金标子集联合报告。
- SmileChat（qiuhuachuan/smile）与本地 5.5 万段高度同源，故未重复全量下载。
- 纯「贴吧热语金标大盘」公开可下载资源少；热语仍可人工扩补。

## 重新采集

```bash
# 仓库根目录；默认镜像
set HF_ENDPOINT=https://hf-mirror.com
python oiu/scripts/ingest_external_student_datasets.py --cpcd-students 10 --max-enriched 20000 --client-only-cpcd
```

## 云端同步策略

- **必传**：`upload_ready_student_enriched.csv` + `README.md` + `SOURCES.md` + `ingest_meta.json` + 热语/校园子集 CSV  
- **选传**：`cpcd_student_ai_turns.csv`（全量轮次，体积大）  
- **不传**：`data/external_raw/`（原始 JSON）

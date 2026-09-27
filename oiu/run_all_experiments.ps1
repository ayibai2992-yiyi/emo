param(
    [string]$PythonExe = "python",
    [string]$CsvPath = ".\data_template.csv",
    [string]$OutputDir = ".\results",
    [string]$Encoding = "utf-8",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Seed = 42,
    [string]$UserCol = "",
    [string]$TextCol = "",
    [string]$LabelCol = "",
    [string]$CrisisCol = "",
    [string]$HistoryCol = "",
    [string]$HistorySep = "\n",
    [double]$Layer2ThresholdFactor = 1.0,
    [Nullable[double]]$FixedThreshold = $null,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Title,
        [string]$Command
    )
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host ("[步骤] {0}" -f $Title) -ForegroundColor Cyan
    Write-Host ("命令: {0}" -f $Command) -ForegroundColor DarkGray
    Write-Host ("=" * 72) -ForegroundColor DarkCyan

    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw ("步骤失败: {0}" -f $Title)
    }
}

try {
    Set-Location -Path $PSScriptRoot

    Write-Host "当前目录: $PSScriptRoot" -ForegroundColor DarkGray
    Write-Host "Python: $PythonExe" -ForegroundColor DarkGray
    Write-Host "设备: $Device, 随机种子: $Seed" -ForegroundColor DarkGray
    Write-Host "CSV: $CsvPath" -ForegroundColor DarkGray

    if (-not $SkipInstall) {
        Invoke-Step `
            -Title "安装基础依赖（tests\requirements.txt）" `
            -Command "$PythonExe -m pip install -r tests\requirements.txt"
        Invoke-Step `
            -Title "安装深度学习依赖（torch + transformers）" `
            -Command "$PythonExe -m pip install torch transformers"
    } else {
        Write-Host "已跳过依赖安装（-SkipInstall）" -ForegroundColor Yellow
    }

    if (-not (Test-Path -Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    $realEvalJson = Join-Path $OutputDir "real_eval_metrics.json"
    $protocolLog = Join-Path $OutputDir "protocol_demo.log"
    $ablationLog = Join-Path $OutputDir "ablation_suite.log"
    $baselineJson = Join-Path $OutputDir "baseline_comparison.json"
    $paperSummaryPrefix = Join-Path $OutputDir "paper_result_summary"

    Invoke-Step `
        -Title "实验 1/4：协议演示（run_protocol_demo.py）" `
        -Command "$PythonExe experiments\run_protocol_demo.py | Tee-Object -FilePath `"$protocolLog`""

    if (-not (Test-Path -Path $CsvPath)) {
        throw ("未找到 CSV 文件: {0}`n请先编辑 data_template.csv 或传入 -CsvPath。" -f $CsvPath)
    }

    $evalArgs = @(
        "experiments\run_real_dataset_eval.py",
        "--csv", "`"$CsvPath`"",
        "--encoding", $Encoding,
        "--device", $Device,
        "--seed", $Seed,
        "--output-json", "`"$realEvalJson`"",
        "--history-sep", "`"$HistorySep`"",
        "--layer2-threshold-factor", $Layer2ThresholdFactor
    )

    if ($UserCol.Trim()) { $evalArgs += @("--user-col", $UserCol) }
    if ($TextCol.Trim()) { $evalArgs += @("--text-col", $TextCol) }
    if ($LabelCol.Trim()) { $evalArgs += @("--label-col", $LabelCol) }
    if ($CrisisCol.Trim()) { $evalArgs += @("--crisis-col", $CrisisCol) }
    if ($HistoryCol.Trim()) { $evalArgs += @("--history-col", $HistoryCol) }
    if ($FixedThreshold -ne $null) { $evalArgs += @("--fixed-threshold", $FixedThreshold.Value) }

    $realEvalCommand = "$PythonExe " + ($evalArgs -join " ")

    Invoke-Step `
        -Title "实验 2/4：真实数据评估（run_real_dataset_eval.py）" `
        -Command $realEvalCommand

    $baselineArgs = @(
        "experiments\run_baseline_comparison.py",
        "--csv", "`"$CsvPath`"",
        "--encoding", $Encoding,
        "--device", $Device,
        "--seed", $Seed,
        "--history-sep", "`"$HistorySep`"",
        "--layer2-threshold-factor", $Layer2ThresholdFactor,
        "--output-json", "`"$baselineJson`""
    )
    if ($UserCol.Trim()) { $baselineArgs += @("--user-col", $UserCol) }
    if ($TextCol.Trim()) { $baselineArgs += @("--text-col", $TextCol) }
    if ($LabelCol.Trim()) { $baselineArgs += @("--label-col", $LabelCol) }
    if ($CrisisCol.Trim()) { $baselineArgs += @("--crisis-col", $CrisisCol) }
    if ($HistoryCol.Trim()) { $baselineArgs += @("--history-col", $HistoryCol) }
    $baselineCommand = "$PythonExe " + ($baselineArgs -join " ")

    Invoke-Step `
        -Title "实验 3/4：基线对比（run_baseline_comparison.py）" `
        -Command $baselineCommand

    Invoke-Step `
        -Title "实验 4/4：消融实验（run_ablation_suite.py）" `
        -Command "$PythonExe experiments\run_ablation_suite.py | Tee-Object -FilePath `"$ablationLog`""

    Invoke-Step `
        -Title "导出论文摘要（export_paper_summary.py）" `
        -Command "$PythonExe experiments\export_paper_summary.py --input-json `"$realEvalJson`" --output-prefix `"$paperSummaryPrefix`""

    Write-Host ""
    Write-Host "全部实验已完成。" -ForegroundColor Green
    Write-Host ("真实评估指标: {0}" -f $realEvalJson) -ForegroundColor Green
    Write-Host ("基线对比指标: {0}" -f $baselineJson) -ForegroundColor Green
    Write-Host ("协议演示日志: {0}" -f $protocolLog) -ForegroundColor Green
    Write-Host ("消融实验日志: {0}" -f $ablationLog) -ForegroundColor Green
    Write-Host ("论文摘要(中): {0}" -f ($paperSummaryPrefix + ".zh.txt")) -ForegroundColor Green
    Write-Host ("论文摘要(英): {0}" -f ($paperSummaryPrefix + ".en.txt")) -ForegroundColor Green
    Write-Host ("论文摘要(MD): {0}" -f ($paperSummaryPrefix + ".md")) -ForegroundColor Green
    Write-Host ""
    Write-Host "你可以通过参数映射任意列名（-UserCol/-TextCol/-LabelCol 等）。" -ForegroundColor Yellow
    Write-Host "示例：" -ForegroundColor Yellow
    Write-Host ".\run_all_experiments.ps1 -CsvPath .\your.csv -UserCol uid -TextCol content -LabelCol emotion_id -CrisisCol is_crisis -HistoryCol context"
}
catch {
    Write-Host ""
    Write-Host "实验中断：" -ForegroundColor Red -NoNewline
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

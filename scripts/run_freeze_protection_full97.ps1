param(
    [string]$ConfigDir = "config/generated/freeze_protection_full97",
    [string]$RunRoot = "results/backtests",
    [string]$BatchLog = "logs/freeze_protection_full97_batch.log",
    [string]$WaitForRunDir = "",
    [int]$ReviewMaxPairs = 97,
    [int]$ReviewMaxPoints = 2000
)

$ErrorActionPreference = "Stop"
$projectRoot = (Get-Location).Path
$logPath = Join-Path $projectRoot $BatchLog
$logParent = Split-Path -Parent $logPath
New-Item -ItemType Directory -Force -Path $logParent | Out-Null

function Write-BatchLog([string]$Message) {
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Wait-ForReport([string]$RunDir) {
    if ([string]::IsNullOrWhiteSpace($RunDir)) {
        return
    }
    $reportDir = Join-Path $projectRoot $RunDir
    $overview = Join-Path $reportDir "trade_review.html"
    $summary = Join-Path $reportDir "portfolio_summary.png"
    $errorFile = Join-Path $reportDir "trade_review_error.txt"
    Write-BatchLog "等待当前运行完成: $reportDir"
    while (
        (!(Test-Path -LiteralPath $overview) -or !(Test-Path -LiteralPath $summary)) -and
        !(Test-Path -LiteralPath $errorFile)
    ) {
        Start-Sleep -Seconds 30
    }
    if (Test-Path -LiteralPath $errorFile) {
        Write-BatchLog "当前运行报告失败，继续后续配置: $errorFile"
    } else {
        Write-BatchLog "当前运行已完成: $reportDir"
    }
}

Wait-ForReport $WaitForRunDir

$configs = Get-ChildItem -LiteralPath (Join-Path $projectRoot $ConfigDir) -Filter "*.yaml" -File | Sort-Object Name
if ($configs.Count -eq 0) {
    throw "没有找到配置: $ConfigDir"
}

$failed = [System.Collections.Generic.List[string]]::new()
$total = $configs.Count
for ($index = 0; $index -lt $total; $index++) {
    $config = $configs[$index]
    $raw = Get-Content -LiteralPath $config.FullName -Raw
    $runName = if ($raw -match '(?m)^\s*run_name:\s*(\S+)\s*$') { $Matches[1] } else { $config.BaseName }
    $runDir = Join-Path $projectRoot (Join-Path $RunRoot $runName)
    $overview = Join-Path $runDir "trade_review.html"
    $summary = Join-Path $runDir "portfolio_summary.png"
    if ((Test-Path -LiteralPath $overview) -and (Test-Path -LiteralPath $summary)) {
        Write-BatchLog "[$($index + 1)/$total] 已存在，跳过: $runName"
        continue
    }

    $runLog = Join-Path $projectRoot (Join-Path "logs/freeze_protection_full97" "$runName.log")
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runLog) | Out-Null
    Write-BatchLog "[$($index + 1)/$total] 开始: $runName"
    try {
        & python scripts/run_backtest.py $config.FullName `
            --review-max-points $ReviewMaxPoints `
            --review-max-pairs $ReviewMaxPairs *>&1 | Tee-Object -FilePath $runLog
        if ($LASTEXITCODE -ne 0) {
            throw "退出码 $LASTEXITCODE"
        }
        if (!(Test-Path -LiteralPath $overview) -or !(Test-Path -LiteralPath $summary)) {
            throw "回测进程结束但报告文件不完整"
        }
        Write-BatchLog "[$($index + 1)/$total] 完成: $runName"
    } catch {
        $failed.Add($runName)
        Write-BatchLog "[$($index + 1)/$total] 失败: $runName -- $($_.Exception.Message)"
    }
}

if ($failed.Count -eq 0) {
    Write-BatchLog "全部 $total 个 Freeze 止盈止损回测完成"
} else {
    Write-BatchLog "批处理完成，但失败 $($failed.Count) 个: $($failed -join ', ')"
    exit 1
}

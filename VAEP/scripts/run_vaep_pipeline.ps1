# VAEP Pipeline 전체 실행 스크립트
# PowerShell에서 실행

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VAEP Pipeline - Full Execution" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# 현재 위치 확인
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "[Step 1/3] Data Preprocessing" -ForegroundColor Yellow
Write-Host "Preprocessing Wyscout events data..." -ForegroundColor Gray
python preprocess_wyscout.py --log_file ../logs/preprocess.log

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in preprocessing! Check logs/preprocess.log" -ForegroundColor Red
    exit 1
}

Write-Host "Preprocessing completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 2/3] Model Training" -ForegroundColor Yellow
Write-Host "Training VAEP model..." -ForegroundColor Gray
python train_vaep_model.py --log_file ../logs/train.log --epochs 30

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in training! Check logs/train.log" -ForegroundColor Red
    exit 1
}

Write-Host "Training completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "[Step 3/3] Player VAEP Computation" -ForegroundColor Yellow
Write-Host "Computing player VAEP..." -ForegroundColor Gray
python compute_player_vaep.py --log_file ../logs/compute.log

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in computation! Check logs/compute.log" -ForegroundColor Red
    exit 1
}

Write-Host "VAEP computation completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Pipeline completed!" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files:" -ForegroundColor White
Write-Host "  1. data/processed/vaep_train_events.csv" -ForegroundColor Gray
Write-Host "  2. data/processed/vaep_eval_events_england.csv" -ForegroundColor Gray
Write-Host "  3. data/models/vaep_model.pt" -ForegroundColor Gray
Write-Host "  4. data/models/vaep_config.json" -ForegroundColor Gray
Write-Host "  5. data/vaep_results/player_match_vaep_england.csv" -ForegroundColor Gray
Write-Host "  6. data/vaep_results/player_season_vaep_england.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "Check logs folder for detailed execution logs." -ForegroundColor Yellow

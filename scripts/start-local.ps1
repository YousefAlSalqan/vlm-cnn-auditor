param([switch]$Check, [switch]$RestartQwen)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements-cpu.lock first. See docs/V1_GUIDE.md.' }
if (-not (Test-Path -LiteralPath '.env')) { throw 'Copy .env.example to .env and configure PostgreSQL first.' }

# Read known settings as data. Never evaluate .env or display database credentials.
$configCode = @'
import json, os
from dotenv import dotenv_values
from auditor.config import Settings
settings = Settings()
values = dotenv_values('.env')
names = ('POSTGRES_BIN', 'POSTGRES_DATA', 'POSTGRES_PORT', 'OLLAMA_EXE', 'OLLAMA_MODELS')
print(json.dumps({name: os.environ.get(name, values.get(name)) for name in names} | {
    'ollama_url': settings.ollama_base_url, 'server_port': settings.server_port,
    'weights': str(settings.weights_path), 'data_root': str(settings.data_root)}))
'@
$configText = & $python -c $configCode
if ($LASTEXITCODE -ne 0) { throw 'Configuration is invalid. Check .env without sharing its credentials.' }
$config = $configText | ConvertFrom-Json
$ollamaUri = [Uri]$config.ollama_url
$logRoot = Join-Path $projectRoot '.local-runtime'

function Test-LocalPort([string]$HostName, [int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        return $task.Wait(1000) -and $client.Connected
    } catch { return $false } finally { $client.Dispose() }
}

if (-not $Check) {
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    if ($config.POSTGRES_BIN -and $config.POSTGRES_DATA) {
        $pgCtl = Join-Path $config.POSTGRES_BIN 'pg_ctl.exe'
        if (-not (Test-Path -LiteralPath $pgCtl) -or -not (Test-Path -LiteralPath $config.POSTGRES_DATA)) {
            throw 'POSTGRES_BIN or POSTGRES_DATA does not exist. This launcher does not initialize databases.'
        }
        $pgPort = if ($config.POSTGRES_PORT) { [int]$config.POSTGRES_PORT } else { 55432 }
        if ($pgPort -lt 1024 -or $pgPort -gt 65535) { throw 'POSTGRES_PORT must be 1024 through 65535.' }
        & $pgCtl status -D $config.POSTGRES_DATA 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            & $pgCtl start -D $config.POSTGRES_DATA -l (Join-Path $logRoot 'postgres.log') -o "-h 127.0.0.1 -p $pgPort" -w -t 15
            if ($LASTEXITCODE -ne 0) { throw 'Local PostgreSQL did not start. Check .local-runtime/postgres.log.' }
        }
    }
    $ollamaPidFile = Join-Path $logRoot 'ollama.pid'
    if ($RestartQwen) {
        if (-not $config.OLLAMA_EXE -or -not (Test-Path -LiteralPath $ollamaPidFile)) {
            throw 'Restart your existing Ollama manually. Only a launcher-owned portable process can be restarted here.'
        }
        $ownedPid = [int](Get-Content -LiteralPath $ollamaPidFile -Raw).Trim()
        $ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$ownedPid"
        if ($ownedProcess) {
            if ($ownedProcess.ExecutablePath -ne [IO.Path]::GetFullPath($config.OLLAMA_EXE) -or $ownedProcess.CommandLine -notmatch '\bserve\s*$') {
                throw 'Saved Ollama PID belongs to another process. It was not stopped.'
            }
            Stop-Process -Id $ownedPid
            Wait-Process -Id $ownedPid -Timeout 10 -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-LocalPort $ollamaUri.Host $ollamaUri.Port)) {
        if (-not $config.OLLAMA_EXE -or -not (Test-Path -LiteralPath $config.OLLAMA_EXE)) {
            throw 'Start local Ollama first, or set OLLAMA_EXE to a portable installation.'
        }
        $env:OLLAMA_HOST = $ollamaUri.Authority
        $env:OLLAMA_NO_CLOUD = 'true'
        $env:OLLAMA_NUM_PARALLEL = '1'
        if ($config.OLLAMA_MODELS) { $env:OLLAMA_MODELS = $config.OLLAMA_MODELS }
        $service = Start-Process -FilePath $config.OLLAMA_EXE -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'ollama.stdout.log') -RedirectStandardError (Join-Path $logRoot 'ollama.stderr.log')
        Set-Content -LiteralPath $ollamaPidFile -Value $service.Id
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        while (-not (Test-LocalPort $ollamaUri.Host $ollamaUri.Port)) {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Ollama did not start. Check .local-runtime/ollama.stderr.log.' }
            Start-Sleep -Milliseconds 250
        }
    }
}

$checkCode = @'
from pathlib import Path
import shutil
from auditor.config import Settings
from auditor.db import create_engine_for, require_schema
from auditor.models import OllamaReviewer
settings = Settings()
assert shutil.which(settings.ffprobe_binary), 'Install FFmpeg and add ffprobe to PATH'
assert settings.weights_path.is_file(), 'Download YOLO weights: python -m auditor.models --download-yolo'
engine = create_engine_for(settings)
try:
    require_schema(engine)
finally:
    engine.dispose()
identity = OllamaReviewer(settings.ollama_base_url, settings.ollama_model).identity()
assert identity.get('model_digest'), 'Local Qwen model unavailable. Install qwen3-vl:2b-instruct in this Ollama runtime'
print('Ready: PostgreSQL schema, FFprobe, YOLO checkpoint and local Qwen model.')
'@
& $python -c $checkCode
if ($LASTEXITCODE -ne 0) { throw 'Readiness check failed. See docs/V1_GUIDE.md for setup and recovery.' }
if ($Check) { return }
$address = 'http://127.0.0.1:' + $config.server_port
if (Test-LocalPort '127.0.0.1' $config.server_port) {
    Write-Host "A process already uses $address. Open it if it is your auditor, or choose another SERVER_PORT."
    return
}
Write-Host "Open $address. Press Ctrl+C here to stop the app."
& $python -m auditor.app
if ($LASTEXITCODE -ne 0) { throw 'Auditor stopped with an error. See the message above.' }

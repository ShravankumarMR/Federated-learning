$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

streamlit run src/app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

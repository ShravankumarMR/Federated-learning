$ErrorActionPreference = "Stop"

if (-Not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

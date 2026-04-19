@echo off
setlocal
cd /d "%~dp0"
start "Flask Backend" cmd /k "cd /d %~dp0 && python app.py"
echo Waiting for backend health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 90;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health -TimeoutSec 2; if($r.StatusCode -eq 200){$ok=$true; break} } catch {}; Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo Backend did not become healthy in time.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:5000/api/health"

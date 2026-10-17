@echo off
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)
%PY% -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo If dependencies are missing, first run:
  echo   %PY% -m pip install -r requirements.txt
  pause
)

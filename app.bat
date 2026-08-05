@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run

echo Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment. Is Python installed and on PATH?
    exit /b 1
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

echo Ensuring polars-lts-cpu (not plain polars) is active...
".venv\Scripts\python.exe" -m pip uninstall -y polars
".venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps polars-lts-cpu

echo Applying Django migrations...
".venv\Scripts\python.exe" webui\manage.py migrate

:run
echo Starting server on 0.0.0.0:8000 (reachable at http://100.121.87.117:8000 and http://127.0.0.1:8000) ...
start "Django dev server" ".venv\Scripts\python.exe" webui\manage.py runserver 0.0.0.0:8000

echo Waiting for the server to come up (this can take a while on a cold start --
echo views.py pulls in pandas/scikit-learn/linopy at Django startup)...
set RETRIES=180
:waitloop
curl -s -o nul "http://127.0.0.1:8000/"
if not errorlevel 1 goto :serverup
set /a RETRIES-=1
if %RETRIES% leq 0 (
    echo Server did not respond after 6 minutes -- check the "Django dev server" console
    echo window for errors. Not opening the browser.
    goto :end
)
ping 127.0.0.1 -n 3 >nul
goto :waitloop

:serverup
echo Server is up.
start "" "http://127.0.0.1:8000/"

:end

endlocal

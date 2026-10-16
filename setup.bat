@echo off
REM Cai dat moi truong Windows
py -3.11 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Kiem tra FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [THIEU] FFmpeg chua co. Chay: winget install Gyan.FFmpeg
) else (
  ffmpeg -version | findstr /C:"ffmpeg version"
)
echo Xong. Chay run.bat de mo ung dung.
pause

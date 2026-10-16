@echo off
REM Mo ung dung desktop
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python main.py
pause

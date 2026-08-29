@echo off
title MJBot - Gold Signal Bot
cd /d "C:\Users\Kiongozi Legit\Downloads\websites\mjbot"
chcp 65001 >nul 2>&1

:restart
echo [%date% %time%] Starting bot...
python -u bot.py >> bot.log 2>&1
echo [%date% %time%] Bot exited (code %ERRORLEVEL%). Restarting in 10 seconds...
timeout /t 10 /nobreak >nul
goto restart

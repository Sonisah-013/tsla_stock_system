@echo off
title Tesla Predictor Server
:: Kill any 'ghost' process on 8081 first
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8081') do taskkill /f /pid %%a
echo ✅ Port 8081 Cleared.
echo 🚀 Starting Tesla Predictor Dashboard...
:: Change 'main:app' to the name of your python file
uvicorn main:app --host 127.0.0.1 --port 8081 --reload
pause

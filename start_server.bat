@echo off
title ASTRAQUANT X-2026: Flash Crash Risk Management Terminal
echo ======================================================================
echo Starting AstraQuant PS-13 Full-Stack Application...
echo Access URL: http://127.0.0.1:8000
echo ======================================================================
"C:\socratic_tutor\venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
pause

@echo off
set PY=C:\Users\Schub\.workbuddy\binaries\python\versions\3.13.12\python.exe
set PROJ=%~dp0
set PYTHONPATH=%PROJ%src;%PYTHONPATH%
"%PY%" -m pytest -q "%PROJ%tests"

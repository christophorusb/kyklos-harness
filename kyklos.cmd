@echo off
rem Launch Kyklos with this folder's venv, from any directory.
"%~dp0.venv\Scripts\python.exe" "%~dp0kyklos.py" %*

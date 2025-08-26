@echo off
REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python first.
    exit /b 1
)

REM 检查 venv 目录是否存在
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

REM 更新 pip
python -m pip install --upgrade pip

REM 安装依赖
echo Installing requirements...
pip install -r requirements.txt

echo Virtual environment is ready!
echo To activate the environment, run: .venv\Scripts\activate.bat

@echo off
chcp 65001 >nul
title 手势识别 Web 应用

echo =================================================
echo   手势数字识别 Web 应用
echo   基于 CNN 模型 (PyTorch)
echo =================================================
echo.

:: 使用 Python 3.11（已安装 PyTorch 2.12）
set PYTHON_PATH=G:\Users\dell\AppData\Local\Programs\Python\Python311\python.exe

:: 检查 Python 是否存在
if not exist "%PYTHON_PATH%" (
    echo [错误] 找不到 Python: %PYTHON_PATH%
    echo [提示] 请检查 Python 安装路径
    pause
    exit /b 1
)

:: 检查 torch 是否可用
"%PYTHON_PATH%" -c "import torch; print('PyTorch版本:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo [错误] PyTorch 未安装或导入失败
    pause
    exit /b 1
)

:: 检查模型文件
if not exist "models\model.pkl" (
    echo [错误] 模型文件不存在: models\model.pkl
    echo [提示] 请先运行 train.py 训练模型
    pause
    exit /b 1
)

echo [启动] 正在启动服务器...
echo [启动] 请稍候，浏览器将自动打开...
echo.

:: 启动 Flask 服务器
start "" "http://localhost:5000"
"%PYTHON_PATH%" app.py

:: 如果到这里说明服务器已关闭
echo.
echo [提示] 服务器已关闭
pause

@echo off
chcp 65001 >nul

:: 设置Python解释器路径
set PYTHON_PATH=%~dp0py311\python.exe

:: 检查Python解释器是否存在
if not exist "%PYTHON_PATH%" (
    echo Python解释器未找到: %PYTHON_PATH%
    echo 请确保py311文件夹在批处理文件的同一目录下。
    pause
    exit /b 1
)

:: 激活虚拟环境（如果存在）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo 未找到虚拟环境,将直接使用系统Python。
)

:: 安装requirements.txt中的依赖
echo 正在安装依赖...
"%PYTHON_PATH%" -m pip install --upgrade pip
"%PYTHON_PATH%" -m pip install -r requirements.txt
"%PYTHON_PATH%" -m pip install dill

:: 安装特定版本的PyTorch（根据CUDA版本）
echo 正在安装PyTorch...
"%PYTHON_PATH%" -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo 依赖安装完成。
pause
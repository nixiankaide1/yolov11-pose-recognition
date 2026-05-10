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

:: 运行Python脚本来测试CUDA
"%PYTHON_PATH%" -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else '不适用'); print('GPU数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('当前GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

pause
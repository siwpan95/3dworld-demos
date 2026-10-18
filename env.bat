@echo off
REM 3dworld-demos 环境变量一键设置（2026-10-19 建）
REM 用法：在本仓根目录执行 env.bat，然后跑演示
REM
REM 两个变量的作用：
REM   WORLD3D_PROJECT_ROOT  指向本仓根 —— --shot 截图落点（docs/）靠它定位
REM   WORLD3D_CONFIGS       指向本仓 configs/ —— cordis 配置树靠它定位
REM （pip 安装的引擎推导不出本仓路径，故必须显式设置）

set "WORLD3D_PROJECT_ROOT=%~dp0"
if "%WORLD3D_PROJECT_ROOT:~-1%"=="\" set "WORLD3D_PROJECT_ROOT=%WORLD3D_PROJECT_ROOT:~0,-1%"
set "WORLD3D_CONFIGS=%WORLD3D_PROJECT_ROOT%\configs"

echo [env] WORLD3D_PROJECT_ROOT=%WORLD3D_PROJECT_ROOT%
echo [env] WORLD3D_CONFIGS=%WORLD3D_CONFIGS%
echo [env] 已设置。现在可以运行：python examples\demoscene_tunnel.py --smoke

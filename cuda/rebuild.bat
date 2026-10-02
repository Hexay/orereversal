@echo off
rem Windows build of oretest, matcher and veintest (Linux: build.sh). ARCH defaults to sm_89 (RTX 40-series).
setlocal
cd /d "%~dp0"
if "%ARCH%"=="" set ARCH=sm_89
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -find VC\Auxiliary\Build\vcvars64.bat`) do set "VCVARS=%%i"
if not defined VCVARS (echo vcvars64.bat not found: install VS 2022 Build Tools with the C++ workload & exit /b 1)
if not defined CUDA_PATH (echo CUDA_PATH not set: install the CUDA Toolkit & exit /b 1)
call "%VCVARS%" >nul
set "PATH=%CUDA_PATH%\bin;%PATH%"
echo === building oretest (host) ===
cl /nologo /O2 /fp:strict oretest.c /Fe:oretest.exe >nul
set OT=%ERRORLEVEL%
echo oretest_exit=%OT%
echo === building matcher (device, %ARCH%) ===
nvcc -O2 -std=c++17 -arch=%ARCH% -Xcompiler /openmp matcher.cu -o matcher.exe
set MT=%ERRORLEVEL%
echo matcher_exit=%MT%
echo === building veintest (device, %ARCH%) ===
nvcc -O2 -std=c++17 -arch=%ARCH% veintest.cu -o veintest.exe >nul
set VT=%ERRORLEVEL%
echo veintest_exit=%VT%
if not "%OT%%MT%%VT%"=="000" exit /b 1

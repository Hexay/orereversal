@echo off
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
echo === building oretest (host) ===
cl /nologo /O2 /fp:strict oretest.c /Fe:oretest.exe >nul
set OT=%ERRORLEVEL%
echo oretest_exit=%OT%
echo === building matcher (device) ===
nvcc -O2 -std=c++17 -arch=sm_89 -Xcompiler /openmp -Xptxas -v matcher.cu -o matcher.exe
set MT=%ERRORLEVEL%
echo matcher_exit=%MT%
echo === building veintest (device) ===
nvcc -O2 -std=c++17 -arch=sm_89 veintest.cu -o veintest.exe >nul
set VT=%ERRORLEVEL%
echo veintest_exit=%VT%
if not "%OT%%MT%%VT%"=="000" exit /b 1

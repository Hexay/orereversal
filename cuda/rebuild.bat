@echo off
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
echo === building oretest (host) ===
cl /nologo /O2 /fp:strict oretest.c /Fe:oretest.exe >nul
echo oretest_exit=%ERRORLEVEL%
echo === building matcher (device) ===
nvcc -O2 -std=c++17 -arch=sm_89 -Xcompiler /openmp -Xptxas -v matcher.cu -o matcher.exe
echo matcher_exit=%ERRORLEVEL%

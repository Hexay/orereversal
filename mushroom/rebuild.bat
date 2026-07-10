@echo off
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
echo === building biometest (host) ===
cl /nologo /O2 /fp:strict biometest.c /Fe:biometest.exe >nul
echo biometest_exit=%ERRORLEVEL%
echo === building finder (device) ===
nvcc -O2 -arch=sm_89 -Xcompiler /openmp -Xptxas -v finder.cu -o finder.exe
echo finder_exit=%ERRORLEVEL%

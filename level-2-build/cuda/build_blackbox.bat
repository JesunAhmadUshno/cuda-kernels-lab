@echo off
rem Build and run the CUDA Build harness on BlackBox (the fleet GPU node).
rem Requires: CUDA toolkit (nvcc) and the VS 18 BuildTools host compiler.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" -O3 -std=c++17 -o kernels.exe kernels.cu
if errorlevel 1 exit /b 1
kernels.exe

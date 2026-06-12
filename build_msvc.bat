@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
nvcc -arch=sm_120 -Iinclude --use_fast_math -Xptxas -O3 -lineinfo -maxrregcount=128 -shared -o lib/minesweeper.dll src/minesweeper_kernel.cu
if %errorlevel% equ 0 (
    echo BUILD SUCCESS
) else (
    echo BUILD FAILED
)

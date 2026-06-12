@echo off
REM CUDA Minesweeper - Build script for Windows
REM Compiles CUDA kernel into a shared DLL library

echo Building CUDA Minesweeper kernel...
echo.

REM Check if nvcc is available
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit 12.8+
    echo Current GPU requires Compute Capability 12.0 support
    pause
    exit /b 1
)

REM Create lib directory if not exists
if not exist lib mkdir lib

REM Compile with optimization flags
nvcc -arch=sm_120 -Iinclude --use_fast_math -Xptxas -O3 -lineinfo -maxrregcount=128 -shared -o lib/minesweeper.dll src/minesweeper_kernel.cu

if %errorlevel% equ 0 (
    echo.
    echo Build successful! Output: lib/minesweeper.dll
    echo.
) else (
    echo.
    echo Build failed!
    echo.
    pause
    exit /b 1
)

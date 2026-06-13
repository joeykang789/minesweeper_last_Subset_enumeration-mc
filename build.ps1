# CUDA Minesweeper - Build script for Windows PowerShell
# Sets up Visual Studio environment and compiles CUDA kernel

$ErrorActionPreference = "Stop"

Write-Host "Building CUDA Minesweeper kernel..." -ForegroundColor Cyan
Write-Host ""

# Check nvcc
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Host "ERROR: nvcc not found in PATH. Please install CUDA Toolkit 13.0+ (for CC 12.0)" -ForegroundColor Red
    exit 1
}

# Setup Visual Studio environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$devShellDll = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"

if (Test-Path $devShellDll) {
    Import-Module $devShellDll -ErrorAction SilentlyContinue
    Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64" -SkipAutomaticLocation 2>$null
    Write-Host "Visual Studio 2022 environment loaded." -ForegroundColor Green
} else {
    # Try to find VS installation path dynamically
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsPath) {
            $devShellDll = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
            Import-Module $devShellDll -ErrorAction SilentlyContinue
            Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64" -SkipAutomaticLocation 2>$null
            Write-Host "Visual Studio environment loaded from: $vsPath" -ForegroundColor Green
        }
    }
}

# Verify cl.exe is available
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Host "ERROR: cl.exe (MSVC compiler) not found." -ForegroundColor Red
    Write-Host "Please install Visual Studio with 'Desktop development with C++' workload." -ForegroundColor Red
    exit 1
}

# Create lib directory
if (-not (Test-Path "lib")) {
    New-Item -ItemType Directory -Path "lib" | Out-Null
}

# Compile CUDA kernel
$includes = "-Iinclude"
$arch = "-arch=sm_120"
$output = "lib/minesweeper.dll"
$source = "src/minesweeper_kernel.cu"

Write-Host "Compiling: $source" -ForegroundColor Yellow
Write-Host "Output:    $output" -ForegroundColor Yellow
Write-Host "Arch:      $arch" -ForegroundColor Yellow
Write-Host ""

nvcc $arch $includes "--use_fast_math" -Xptxas -O3 -lineinfo -maxrregcount=128 -Xlinker "/DEF:minesweeper.def" -shared -o $output $source

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build successful! Output: $output" -ForegroundColor Green
    Write-Host ""
    
    # Show file size
    $fileSize = (Get-Item $output).Length
    Write-Host "DLL size: $([math]::Round($fileSize / 1KB, 1)) KB" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

[CmdletBinding()]
param(
    [string]$BuildDirectory,
    [switch]$CpuOnly,
    [switch]$Validate,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskVsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath $taskVsWhere)) {
        throw "MSVC is unavailable. Install Visual Studio C++ tools or run in an x64 Native Tools command prompt."
    }
    # Prefer VS 2022 for broad CUDA host compiler compatibility.
    $taskVsInstall = & $taskVsWhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $taskVsInstall) {
        $taskVsInstall = & $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $taskVsInstall) { throw "No Visual Studio C++ toolchain found." }
    $taskVcvars = Join-Path $taskVsInstall "VC\Auxiliary\Build\vcvars64.bat"
    $taskEnvironment = & cmd.exe /d /s /c "`"$taskVcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "Visual Studio environment setup failed." }
    foreach ($taskEntry in $taskEnvironment) {
        if ($taskEntry -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    if (-not $taskVsInstall -and (Test-Path -LiteralPath $taskVsWhere)) {
        $taskVsInstall = & $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if ($taskVsInstall) {
        $taskNinjaPath = Join-Path $taskVsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
        if (Test-Path -LiteralPath (Join-Path $taskNinjaPath "ninja.exe")) {
            $env:PATH = $taskNinjaPath + ";" + $env:PATH
        }
    }
}

if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) { throw "CMake 3.24 or newer is required." }
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) { throw "Ninja is required; install it or use CMake with a generator of your choice." }

if (-not $BuildDirectory) {
    $BuildDirectory = if ($CpuOnly) { 'build/cpu' } else { 'build/cuda' }
}
if ([System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $taskBuildPath = $BuildDirectory
} else {
    $taskBuildPath = Join-Path $taskProjectRoot $BuildDirectory
}
$taskCudaOption = if ($CpuOnly) { 'OFF' } else { 'ON' }
# Keep CMake's MSVC include-prefix probe and Ninja's build output in UTF-8.
$taskPreviousCodePage = (chcp.com) -replace '\D', ''
try {
    chcp.com 65001 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not switch the build console to UTF-8." }
    $taskConfigureOptions = @()
    if ($Fresh) { $taskConfigureOptions += '--fresh' }
    & cmake @taskConfigureOptions -S $taskProjectRoot -B $taskBuildPath -G Ninja -DCMAKE_BUILD_TYPE=Release "-DDIFFXPBD_ENABLE_CUDA=$taskCudaOption"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
    & cmake --build $taskBuildPath
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
    if ($Validate) {
        & ctest --test-dir $taskBuildPath --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw "Validation failed." }
    }
} finally {
    if ($taskPreviousCodePage) { chcp.com $taskPreviousCodePage | Out-Null }
}

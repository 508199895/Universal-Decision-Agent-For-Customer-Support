# Dot-source from the project directory: . .\Activate-UdaHub.ps1
# Session-only workaround for the failing Miniconda Windows EXE launcher.
$condaBasePython = 'D:\miniconda3\python.exe'
$udaHubPrefix = 'E:\007.agent\007.project\RAG\ENV\uda-hub'
if (-not (Test-Path -LiteralPath $condaBasePython) -or
    -not (Test-Path -LiteralPath "$udaHubPrefix\python.exe")) {
    throw 'Conda base Python or uda-hub Python is missing.'
}

Import-Module 'D:\miniconda3\shell\condabin\Conda.psm1' -Global -ArgumentList @{ChangePs1 = $false}

# Activation scripts reset CONDA_EXE, so restore the Python entry point
# before and after every invocation of Conda's own PowerShell dispatcher.
function global:Invoke-ProjectConda {
    $env:CONDA_EXE = 'D:\miniconda3\python.exe'
    $env:_CE_M = '-m'
    $env:_CE_CONDA = 'conda'
    try {
        Invoke-Conda @args
    } finally {
        $env:CONDA_EXE = 'D:\miniconda3\python.exe'
        $env:_CE_M = '-m'
        $env:_CE_CONDA = 'conda'
    }
}
Set-Alias -Name conda -Value Invoke-ProjectConda -Scope Global
conda activate $udaHubPrefix
if ($LASTEXITCODE -ne 0 -or $env:CONDA_PREFIX -ne $udaHubPrefix) {
    throw 'uda-hub activation failed.'
}
$activePython = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
if ($activePython -ne "$udaHubPrefix\python.exe") {
    throw "uda-hub interpreter mismatch: $activePython"
}
python -c "import os, sys; from pathlib import Path; assert Path(sys.prefix) == Path(os.environ['CONDA_PREFIX']), 'Python prefix does not match the active Conda environment'"
if ($LASTEXITCODE -ne 0) {
    throw 'uda-hub interpreter verification failed.'
}
Write-Host "Activated uda-hub: $env:CONDA_PREFIX"

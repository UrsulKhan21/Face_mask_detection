$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Python311 {
    try {
        & py -3.11 --version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Face Mask Detection setup"
Write-Host "Project folder: $ProjectRoot"
Write-Host ""

if (-not (Test-Command "py")) {
    Write-Host "Python launcher was not found."
}

if (-not (Test-Python311)) {
    Write-Host "Python 3.11 was not found. Trying to install it with winget..."

    if (-not (Test-Command "winget")) {
        throw "winget is not available. Install Python 3.11 manually from https://www.python.org/downloads/release/python-3119/ and run this script again."
    }

    winget install --id Python.Python.3.11 --source winget --accept-package-agreements --accept-source-agreements

    Write-Host ""
    Write-Host "Refreshing PATH for this terminal..."
    $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$MachinePath;$UserPath"

    if (-not (Test-Python311)) {
        throw "Python 3.11 was installed, but this terminal cannot see it yet. Close PowerShell, open it again in this project folder, and rerun .\setup_python311.ps1"
    }
}

Write-Host "Using:"
py -3.11 --version

if (Test-Path ".venv") {
    Write-Host ".venv already exists. Reusing it."
}
else {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
}

Write-Host "Upgrading pip..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "Installing project requirements..."
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Run the app with:"
Write-Host ".\.venv\Scripts\streamlit.exe run app.py"
Write-Host ""
Write-Host "Or activate the environment first:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "streamlit run app.py"

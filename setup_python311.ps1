$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Creating virtual environment with Python 3.11..."
py -3.11 -m venv .venv

Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing project requirements..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "To activate later, run:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then train with:"
Write-Host "python train.py"
Write-Host ""
Write-Host "Or run the app with:"
Write-Host "streamlit run app.py"

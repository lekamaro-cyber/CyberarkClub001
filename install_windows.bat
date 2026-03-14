@echo off
echo ============================================================
echo   XRP Signal Detector - Installation Windows
echo ============================================================
echo.

:: Verifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe!
    echo.
    echo Telecharge Python depuis: https://www.python.org/downloads/
    echo IMPORTANT: Coche "Add Python to PATH" pendant l'installation!
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte:
python --version
echo.

:: Installer les dependances
echo Installation des dependances...
pip install python-binance pandas numpy
echo.

if %errorlevel% neq 0 (
    echo [ERREUR] Echec installation des dependances
    pause
    exit /b 1
)

echo [OK] Dependances installees!
echo.
echo ============================================================
echo   Installation terminee!
echo ============================================================
echo.
echo   Commandes disponibles:
echo.
echo   1. Analyse seule:
echo      python xrp_signal_binance.py
echo.
echo   2. Simulation (dry-run):
echo      python xrp_signal_binance.py --trade --dry-run --key CLE --secret SECRET --loop
echo.
echo   3. Trading reel:
echo      python xrp_signal_binance.py --trade --key CLE --secret SECRET --quantity 50 --loop
echo.
echo ============================================================
pause

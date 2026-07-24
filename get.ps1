# Bake-off one-line installer for Windows (PowerShell 5.1 or later, no admin needed).
#
#   irm https://raw.githubusercontent.com/jdbarzy/bake-off/HEAD/get.ps1 | iex
#
# Installs Bake-off to %USERPROFILE%\bake-off, starts it now and on every sign-in,
# and prints the address to open. Safe to re-run: it updates the app and keeps your
# settings (start-bakeoff.cmd is yours and is never overwritten).
#
# Prefer a file? Save this script, then:  powershell -ExecutionPolicy Bypass -File get.ps1

$ErrorActionPreference = 'Stop'
$Dest = Join-Path $env:USERPROFILE 'bake-off'
$Zip  = 'https://github.com/jdbarzy/bake-off/archive/HEAD.zip'

Write-Host ''
Write-Host 'Bake-off installer (Windows)'

# ---- prerequisites: any Python 3 on PATH ----
$py = $null
foreach ($c in @('python', 'py')) {
  try {
    $v = & $c --version 2>&1
    if ("$v" -match 'Python 3') { $py = $c; break }
  } catch {}
}
if (-not $py) {
  Write-Host 'Python 3 is needed. Install it from https://www.python.org/downloads/' -ForegroundColor Red
  Write-Host '(tick "Add python.exe to PATH" in its installer), then run this line again.' -ForegroundColor Red
  return
}

# ---- stop a running Bake-off so files replace cleanly (settings are separate) ----
Get-CimInstance Win32_Process -Filter "Name LIKE 'py%'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'bake-off.dashboard.server\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# ---- download + extract over the install folder ----
Write-Host 'Downloading Bake-off...'
$tmp = Join-Path $env:TEMP ('bakeoff-' + [guid]::NewGuid().ToString('n'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$zipfile = Join-Path $tmp 'bake-off.zip'
Invoke-WebRequest -UseBasicParsing -Uri $Zip -OutFile $zipfile
Expand-Archive -Path $zipfile -DestinationPath $tmp
$src = Get-ChildItem -Directory $tmp | Where-Object { $_.Name -like 'bake-off-*' } | Select-Object -First 1
if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest | Out-Null }
Copy-Item -Path (Join-Path $src.FullName '*') -Destination $Dest -Recurse -Force
Remove-Item -Recurse -Force $tmp
Write-Host "  installed to $Dest (your settings, if any, were kept)"

# ---- start-bakeoff.cmd: the user's settings file; created once, never overwritten ----
$startCmd = Join-Path $Dest 'start-bakeoff.cmd'
if (-not (Test-Path $startCmd)) {
@"
@echo off
rem Bake-off runtime settings (yours; kept on updates).
rem 0.0.0.0 = reachable on your network; 127.0.0.1 = this machine only.
set DASH_HOST=0.0.0.0
set DASH_PORT=15600
rem set DASH_AUTH=user:pass    (remove "rem" to require a login)
cd /d "%~dp0"
start "Bake-off" /MIN cmd /c "$py dashboard\server.py >> dashboard\dashboard.log 2>&1"
"@ | Set-Content -Path $startCmd -Encoding ASCII
  Write-Host '  created start-bakeoff.cmd with safe defaults (edit it to change port or add a login)'
}

# ---- uninstall-bakeoff.cmd: stop the app + remove the sign-in autostart; files stay ----
$un = Join-Path $Dest 'uninstall-bakeoff.cmd'
@"
@echo off
rem Undo the Windows install: stop Bake-off and remove the sign-in autostart. Files stay.
rem (py%% not py%: batch eats a lone percent, which would break the process match)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name LIKE 'py%%'\" | Where-Object { `$_.CommandLine -match 'bake-off.dashboard.server\.py' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }"
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\bake-off.cmd" 2>nul
echo Removed. Delete this folder to finish; reinstall any time with the installer line.
"@ | Set-Content -Path $un -Encoding ASCII

# ---- start on sign-in (per-user Startup folder; no admin) ----
$startup = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\bake-off.cmd'
"@echo off`r`ncall `"$startCmd`"" | Set-Content -Path $startup -Encoding ASCII

# ---- start it now ----
& cmd.exe /c $startCmd
Start-Sleep -Seconds 2
$ip = $null
try {
  $ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1).IPAddress
} catch {}

Write-Host ''
Write-Host 'Done - Bake-off is installed and starts when you sign in.' -ForegroundColor Green
Write-Host ''
Write-Host '  Open it:'
Write-Host '    this machine:      http://localhost:15600/'
if ($ip) { Write-Host "    any other device:  http://${ip}:15600/  (allow it if Windows Firewall asks)" }
Write-Host ''
Write-Host '  Next: get some models. On each Linux GPU machine, run:  bash mwboot.sh'
Write-Host '  Update: Settings > General in the app, or run this installer line again.'
Write-Host '  Remove: run uninstall-bakeoff.cmd in the bake-off folder.'

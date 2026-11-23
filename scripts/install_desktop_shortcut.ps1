<#
.SYNOPSIS
  Put an "Interview Copilot" shortcut on your Windows Desktop.

.DESCRIPTION
  The shortcut runs scripts\launch_copilot.bat in a console window (it has to be a console: stopping
  transcription sends Ctrl-Break to the recorder). Double-click it, the browser opens on the menu;
  keep the window open during the call and close it (or press Ctrl-C) to stop everything.
  Run it again any time to recreate the shortcut, e.g. after moving the project folder.

.PARAMETER Session
  Optional bundle id to preselect on the Call screen (a folder name under scripts\inputs\sessions).

.PARAMETER Name
  Optional shortcut name. Default: "Interview Copilot" (or "Interview Copilot - <session>").

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_desktop_shortcut.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_desktop_shortcut.ps1 -Session my_interview
#>
param(
  [string]$Session = "",
  [string]$Name = ""
)
$ErrorActionPreference = "Stop"

if ($Session -notmatch '^[A-Za-z0-9_-]*$') {
  throw "Session id '$Session' may only contain letters, digits, '_' and '-'."
}

$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root "scripts\launch_copilot.bat"
if (-not (Test-Path -LiteralPath $launcher)) {
  throw "Launcher not found: $launcher"
}

if (-not $Name) {
  if ($Session) { $Name = "Interview Copilot - $Session" } else { $Name = "Interview Copilot" }
}

# GetFolderPath follows a Desktop redirected to OneDrive; $HOME\Desktop would not.
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "$Name.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath = $launcher
$lnk.Arguments = $Session
$lnk.WorkingDirectory = $root
$lnk.Description = "Interview copilot hub (loopback only). Keep the window open during a call."
$lnk.WindowStyle = 1
$lnk.Save()

Write-Host "Created $lnkPath"
Write-Host "Double-click it to start the copilot; keep its window open during the call."

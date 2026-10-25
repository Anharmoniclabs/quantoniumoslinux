# Run this script as Administrator to allow mobile devices to connect.
# Right-click -> "Run with PowerShell" (as Admin)
New-NetFirewallRule -DisplayName "QU.AI API (port 8000)" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "QU.AI Web (port 8081)" -Direction Inbound -LocalPort 8081 -Protocol TCP -Action Allow
Write-Host "`nFirewall rules added for ports 8000 and 8081." -ForegroundColor Green
Write-Host "Mobile devices on your LAN can now reach the app."
Read-Host "Press Enter to close"

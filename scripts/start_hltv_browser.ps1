param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 9222,
    [datetime]$StartDate = '2023-09-27',
    [datetime]$EndDate = (Get-Date).Date
)

$ErrorActionPreference = 'Stop'
if ($StartDate.Date -gt $EndDate.Date) {
    throw 'StartDate must not be later than EndDate.'
}

$projectRoot = Split-Path -Parent $PSScriptRoot
# Используется отдельный профиль проекта, а не личный профиль Chrome.
$browserProfile = Join-Path $projectRoot 'data\chrome_cdp_profile\hltv_refresh'
$chromeCandidates = @(
    (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
    (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe')
)
$chromePath = $chromeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $chromePath) { throw 'Google Chrome was not found.' }

$existingListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($existingListeners.Count -gt 0) {
    throw "Port $Port is already in use. Do not launch a second browser; inspect the existing listener first."
}

$startText = $StartDate.ToString('yyyy-MM-dd')
$endText = $EndDate.ToString('yyyy-MM-dd')
$resultsUrl = "https://www.hltv.org/results?startDate=$startText&endDate=$endText&offset=0"
$chromeArguments = @(
    "--remote-debugging-port=$Port",
    '--remote-debugging-address=127.0.0.1',
    ('--user-data-dir="' + $browserProfile + '"'),
    '--no-first-run',
    '--no-default-browser-check',
    $resultsUrl
)

# Окно видно пользователю: проверку сайта он проходит самостоятельно.
Start-Process -FilePath $chromePath -ArgumentList $chromeArguments -WindowStyle Normal | Out-Null
$cdpUrl = "http://127.0.0.1:$Port"
$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $version = Invoke-RestMethod -Uri "$cdpUrl/json/version" -TimeoutSec 1
        if ($version.Browser -and $version.webSocketDebuggerUrl) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
if (-not $ready) {
    throw "Chrome opened without an accessible local endpoint. Close only the project browser and try again."
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
if ($listeners.Count -eq 0 -or @($listeners | Where-Object { $_.LocalAddress -notin @('127.0.0.1', '::1') }).Count) {
    throw 'The debugging endpoint is not restricted to loopback. Close the project browser before continuing.'
}

Write-Output "Browser: $($version.Browser)"
Write-Output "Project profile: $browserProfile"
Write-Output "Local connection: $cdpUrl"
Write-Output 'Open HLTV in THIS window and complete any site check manually.'
Write-Output 'Only after the results appear, run the collector with --cdp-url above.'
Write-Output 'Do not sign in to unrelated services in this profile. Close this window when collection is finished.'

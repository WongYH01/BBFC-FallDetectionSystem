# Pull main into the checkout, rebuild the detection image from it and swap the
# local container onto it, then wait for it to report healthy.
#
# Meant for a weekly Windows scheduled task (README, "Run locally"), so a local
# deployment that stays up for weeks still picks up Promotions: a Promotion is a
# commit on main that repins scripts/fetch_weights.py, and the rebuild fetches
# the new head. Only detection is touched; notification keeps running. Exits
# non-zero, which Task Scheduler shows as the last run result, if the git pull,
# the build, the swap or the health check fails. A failed pull or build leaves
# the running container as it was.
#
#   powershell -NoProfile -File infra\local\update-detection.ps1           # .env, the local profile
#   powershell -NoProfile -File infra\local\update-detection.ps1 -WhatIf   # say what it would do
[CmdletBinding(SupportsShouldProcess)]
param(
    # Compose env file, relative to the repo root.
    [string]$EnvFile = ".env",
    # How long the new container may take to report healthy. The image's
    # HEALTHCHECK has a 60 s start period and a 30 s interval.
    [int]$HealthTimeoutSec = 300,
    [string]$LogPath = "$env:LOCALAPPDATA\bbfc\update-detection.log"
)

$ErrorActionPreference = "Stop"
$service = "detection"

# docker writes progress to stderr; fold it into the output as text so the log
# gets it and Windows PowerShell 5.1 does not turn it into errors.
function Invoke-Compose {
    $ErrorActionPreference = "Continue"
    & docker compose --env-file $EnvFile @args 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "docker compose $args failed (exit $LASTEXITCODE)" }
}

function Invoke-Git {
    $ErrorActionPreference = "Continue"
    & git @args 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "git $args failed (exit $LASTEXITCODE)" }
}

# Queries keep stderr apart, so a compose warning never reads as a value.
function Get-ContainerId {
    $id = & docker compose --env-file $EnvFile ps -q $service
    if ($LASTEXITCODE -ne 0) { throw "docker compose ps failed (exit $LASTEXITCODE): is Docker running?" }
    $id
}

# One field of the running container, or $null when none is running: ps
# lists nothing, or the container stopped before inspect reached it.
function Get-ContainerField([string]$Format) {
    $id = Get-ContainerId
    if (-not $id) { return $null }
    $value = & docker inspect --format $Format $id
    if ($LASTEXITCODE -ne 0 -or $null -eq $value) { return $null }
    "$value".Trim()
}

function Get-ContainerImage { Get-ContainerField "{{.Image}}" }

$notRunning = "$service container is not running: docker compose logs $service"

function Wait-Healthy {
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    do {
        $health = Get-ContainerField "{{.State.Health.Status}}"
        if (-not $health) { throw $notRunning }
        if ($health -eq "healthy") { return }
        if ($health -eq "unhealthy") { throw "$service is unhealthy: docker compose logs $service" }
        Start-Sleep -Seconds 10
    } while ((Get-Date) -lt $deadline)
    throw "$service not healthy after $HealthTimeoutSec s (last: $health)"
}

New-Item -ItemType Directory -Force (Split-Path $LogPath) -WhatIf:$false | Out-Null
Start-Transcript -Path $LogPath -Append -WhatIf:$false | Out-Null
try {
    Set-Location (Resolve-Path "$PSScriptRoot\..\..")
    Invoke-Compose config --quiet    # env files present, compose file valid

    $old = Get-ContainerImage
    Write-Output "running image: $(if ($old) { $old } else { 'none' })"

    # Promotions land on main; a checkout on any other branch would keep
    # rebuilding a head that has fallen behind.
    $branch = (& git rev-parse --abbrev-ref HEAD).Trim()
    if ($branch -ne "main") { throw "checkout is on '$branch', not main: git switch main" }

    if ($PSCmdlet.ShouldProcess($service, "pull main, rebuild the image and recreate the container")) {
        Invoke-Git pull --ff-only
        Invoke-Compose build $service
        # Only a changed image recreates the container.
        Invoke-Compose up -d --no-deps $service

        $new = Get-ContainerImage
        if (-not $new) { throw $notRunning }
        Write-Output $(if ($new -eq $old) { "already on the newest image" } else { "swapped to image: $new" })
        Wait-Healthy
        Write-Output "$service is healthy"

        # The old image is untagged now; remove it so weekly builds do not fill
        # the disk. Best effort: another container may still use it.
        if ($old -and $new -ne $old) {
            $ErrorActionPreference = "Continue"
            & docker image rm $old 2>&1 | ForEach-Object { "$_" }
        }
    }
} catch {
    Write-Output "update failed: $_"
    exit 1
} finally {
    Stop-Transcript | Out-Null
}
exit 0

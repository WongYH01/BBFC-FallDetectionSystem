# Pull the newest detection image from Artifact Registry and swap the local
# container onto it, then wait for it to report healthy.
#
# Meant for a weekly Windows scheduled task (README, "Run locally"), so a local
# deployment that stays up for weeks still picks up Promotions. Only detection
# is touched; notification keeps running. Exits non-zero, which Task Scheduler
# shows as the last run result, if the pull, the swap or the health check fails.
# A failed pull leaves the running container as it was.
#
#   powershell -NoProfile -File infra\local\update-detection.ps1           # .env.cloud
#   powershell -NoProfile -File infra\local\update-detection.ps1 -WhatIf   # say what it would do
[CmdletBinding(SupportsShouldProcess)]
param(
    # Compose env file, relative to the repo root.
    [string]$EnvFile = ".env.cloud",
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

# Queries keep stderr apart, so a compose warning never reads as a value.
function Get-ContainerId {
    $id = & docker compose --env-file $EnvFile ps -q $service
    if ($LASTEXITCODE -ne 0) { throw "docker compose ps failed (exit $LASTEXITCODE): is Docker running?" }
    $id
}

function Get-ContainerImage {
    $id = Get-ContainerId
    if ($id) { (& docker inspect --format "{{.Image}}" $id).Trim() }
}

function Wait-Healthy {
    $id = Get-ContainerId
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    do {
        $health = (& docker inspect --format "{{.State.Health.Status}}" $id).Trim()
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

    if ($PSCmdlet.ShouldProcess($service, "pull the newest image and recreate the container")) {
        Invoke-Compose pull $service
        # The pull already happened; --pull never keeps up from contacting the
        # registry again. Only a changed image recreates the container.
        Invoke-Compose up -d --no-deps --pull never $service

        $new = Get-ContainerImage
        Write-Output $(if ($new -eq $old) { "already on the newest image" } else { "swapped to image: $new" })
        Wait-Healthy
        Write-Output "$service is healthy"

        # The old image is untagged now; remove it so weekly pulls do not fill
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

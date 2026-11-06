param([Parameter(Mandatory=$true)][string]$Run, [switch]$Watch, [int]$Interval=5)
$ErrorActionPreference='Stop'
if ($Interval -lt 1) { throw 'Interval must be at least 1 second' }
$taskRun=(Resolve-Path -LiteralPath $Run).Path
do {
    $taskProgress=Join-Path $taskRun 'progress.json'
    if (Test-Path -LiteralPath $taskProgress) {
        $taskState=Get-Content -LiteralPath $taskProgress -Raw -Encoding utf8 | ConvertFrom-Json
        if ($Watch) { Clear-Host }
        if ($null -ne $taskState.total_images) {
            Write-Host ('Batch PID {0} | {1} | complete={2} | images={3} succeeded={4} failed={5} pending={6}' -f $taskState.pid,$taskState.phase,$taskState.complete,$taskState.total_images,$taskState.succeeded,$taskState.failed,$taskState.pending)
            foreach ($taskChild in $taskState.running) {
                Write-Host ('GPU {0} | child PID {1} | attempt {2} | {3}' -f $taskChild.gpu,$taskChild.pid,$taskChild.attempt,$taskChild.input)
                if (Test-Path -LiteralPath $taskChild.progress) {
                    $taskChildState=Get-Content -LiteralPath $taskChild.progress -Raw -Encoding utf8 | ConvertFrom-Json
                    Write-Host ('  step {0}/{1} | N {2}/{3} | phase {4}' -f $taskChildState.step,$taskChildState.total_steps,$taskChildState.N,$taskChildState.cap,$taskChildState.phase)
                }
            }
        } else {
            Write-Host ('PID {0} | {1}/{2} | complete={3}' -f $taskState.pid,$taskState.stage,$taskState.phase,$taskState.complete)
            Write-Host ('Main steps {0}/{1} | N {2}/{3} | optimizer updates {4} | seconds {5:N1}' -f $taskState.step,$taskState.total_steps,$taskState.N,$taskState.cap,$taskState.optimizer_updates,$taskState.elapsed_seconds)
            if ($null -ne $taskState.last_evaluation) {
                $taskMetrics=$taskState.last_evaluation.metrics
                Write-Host ('Last eval step {0} | PSNR {1:N6} | SSIM {2:N6} | MS-SSIM {3:N6}' -f $taskState.last_evaluation.step,$taskMetrics.PSNR,$taskMetrics.SSIM,$taskMetrics.MS_SSIM)
            }
        }
        if ($taskState.error) { Write-Host ('Error: '+$taskState.error) }
        if ($taskState.complete -or $taskState.phase -in @('failed','interrupted','session_limit')) { break }
    } else { Write-Host 'Waiting for progress.json' }
    if (-not $Watch) { break }
    Start-Sleep -Seconds $Interval
} while ($true)

param(
    [string]$Exe = ".\build_all\gol.exe",
    [string]$Out = "results\sweep_gpu_opt.csv",
    [int[]]$Ns = @(256, 512, 1024, 2048, 4096),
    [int[]]$Blocks = @(32, 64, 128, 256),
    [int]$Generations = 100
)

$ErrorActionPreference = "Stop"

Write-Host "SCRIPT STARTED"
Write-Host "Exe: $Exe"
Write-Host "Out: $Out"
Write-Host "Ns: $Ns"
Write-Host "Blocks: $Blocks"

if (!(Test-Path $Exe)) {
    throw "Executable not found: $Exe"
}

New-Item -ItemType Directory -Force results | Out-Null

& $Exe --csv-header | Set-Content -Encoding utf8 $Out

foreach ($n in $Ns) {
    foreach ($engine in "cuda", "opencl") {
        foreach ($block in $Blocks) {
            foreach ($shared in 0, 1) {


                Write-Host "Running engine=$engine n=$n block=$block shared=$shared"

                $args = @(
                    "--engine", $engine,
                    "-r", $n,
                    "-c", $n,
                    "-g", $Generations,
                    "--block", $block
                )

                if ($shared -eq 1) {
                    $args += "--shared"
                }

                & $Exe @args --csv | Out-File -Append -Encoding utf8 $Out
            }
        }
    }
}

Write-Host "SCRIPT FINISHED"
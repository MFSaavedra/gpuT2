param(
    [string]$Exe = ".\build_all\gol.exe",
    [string]$Out = "results\sweep_scaling.csv",
    [int[]]$Ns = @(256, 512, 1024, 2048, 4096),
    [int[]]$CpuThreads = @(1, 0),
    [int]$Generations = 100,

    [int]$BestCudaBlock = 64,
    [int]$BestCudaShared = 0,

    [int]$BestOpenCLBlock = 256,
    [int]$BestOpenCLShared = 1
)

$ErrorActionPreference = "Stop"

Write-Host "SCRIPT STARTED"
Write-Host "Exe: $Exe"
Write-Host "Out: $Out"
Write-Host "Ns: $Ns"
Write-Host "CPU threads: $CpuThreads"
Write-Host "Best CUDA: block=$BestCudaBlock shared=$BestCudaShared"
Write-Host "Best OpenCL: block=$BestOpenCLBlock shared=$BestOpenCLShared"

if (!(Test-Path $Exe)) {
    throw "Executable not found: $Exe"
}

New-Item -ItemType Directory -Force results | Out-Null

& $Exe --csv-header | Set-Content -Encoding utf8 $Out

foreach ($n in $Ns) {
    foreach ($t in $CpuThreads) {
        Write-Host "Running CPU n=$n threads=$t"

        & $Exe --engine cpu -r $n -c $n -g $Generations -t $t --csv |
            Out-File -Append -Encoding utf8 $Out
    }

    Write-Host "Running CUDA n=$n block=$BestCudaBlock shared=$BestCudaShared"

    $cudaArgs = @(
        "--engine", "cuda",
        "-r", $n,
        "-c", $n,
        "-g", $Generations,
        "--block", $BestCudaBlock
    )

    if ($BestCudaShared -eq 1) {
        $cudaArgs += "--shared"
    }

    & $Exe @cudaArgs --csv |
        Out-File -Append -Encoding utf8 $Out

    Write-Host "Running OpenCL n=$n block=$BestOpenCLBlock shared=$BestOpenCLShared"

    $openclArgs = @(
        "--engine", "opencl",
        "-r", $n,
        "-c", $n,
        "-g", $Generations,
        "--block", $BestOpenCLBlock
    )

    if ($BestOpenCLShared -eq 1) {
        $openclArgs += "--shared"
    }

    & $Exe @openclArgs --csv |
        Out-File -Append -Encoding utf8 $Out
}

Write-Host "SCRIPT FINISHED"    
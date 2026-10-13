param([string]$Split = '', [int]$Limit = 500, [switch]$Energy)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$pythonPath = Join-Path $PWD '.venv\Scripts\python.exe'
function Run-Python {
    param([string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed: $Arguments" }
}
$prepareArgs = @('-m','edgeprep.challenge','prepare','--output','results/challenge_data')
if ($Split) { $prepareArgs += @('--split',$Split) }
Run-Python -Arguments $prepareArgs
if (-not $Split) {
    $Split = 'results/challenge_data/provisional_split.json'
    Write-Warning 'Provisional split: these results are diagnostic and may overlap checkpoint training.'
}
$calibration = 'results/challenge_data/calibration.npy'
$exports = @(
    @{Name='baseline'; Args=@()},
    @{Name='qdq_s8'; Args=@('--calibration',$calibration,'--calibration-count','128')},
    @{Name='qdq_u8'; Args=@('--calibration',$calibration,'--calibration-count','128','--unsigned')},
    @{Name='qdq_s8_256'; Args=@('--calibration',$calibration,'--calibration-count','256')},
    @{Name='qdq_percentile'; Args=@('--calibration',$calibration,'--calibration-count','256','--method','Percentile')},
    @{Name='qdq_protected'; Args=@('--calibration',$calibration,'--protect-ends')},
    @{Name='qdq_pointwise'; Args=@('--calibration',$calibration,'--protect-ends','--protect-depthwise')},
    @{Name='prune05'; Args=@('--prune','0.05','--pruning-scope','uniform')},
    @{Name='prune10'; Args=@('--prune','0.10','--pruning-scope','uniform')},
    @{Name='expansion05'; Args=@('--prune','0.05','--pruning-scope','expansion')},
    @{Name='expansion10'; Args=@('--prune','0.10','--pruning-scope','expansion')}
)
foreach ($item in $exports) {
    Run-Python -Arguments (@('-m','edgeprep.challenge','export','--output',"artifacts/$($item.Name)") + $item.Args)
    $model = if ($item.Name.StartsWith('qdq')) { "artifacts/$($item.Name)/model_int8.onnx" } else { "artifacts/$($item.Name)/model_fp32.onnx" }
    Run-Python -Arguments @('-m','edgeprep.challenge','evaluate','--model',$model,'--split',$Split,'--limit',"$Limit",'--output',"results/challenge/$($item.Name)_accuracy.json")
    foreach ($threads in @(1,4)) {
        $benchArgs = @('-m','edgeprep.benchmark','--model',$model,'--size','224','--threads',"$threads",'--runs','300','--inputs','results/challenge_data/timing_input.npy','--labels','results/challenge_data/timing_label.npy','--output',"results/challenge/$($item.Name)_t$threads.json")
        Run-Python -Arguments $benchArgs
    }
}
if ($Energy) {
    foreach ($name in @('baseline','qdq_s8_256','qdq_pointwise')) {
        $suffix = if ($name -eq 'baseline') { 'fp32' } else { 'int8' }
        Run-Python -Arguments @('-m','edgeprep.benchmark','--model',"artifacts/$name/model_$suffix.onnx",'--size','224','--threads','1','--runs','300','--energy-seconds','20','--output',"results/challenge/$($name)_energy.json")
    }
}
Run-Python -Arguments @('-m','edgeprep.challenge_report')

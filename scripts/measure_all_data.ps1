$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$pythonPath = Join-Path $PWD '.venv\Scripts\python.exe'
function Run-Python {
    param([string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed: $Arguments" }
}
Run-Python -Arguments @('-m','edgeprep.challenge','prepare')
Run-Python -Arguments @('-m','edgeprep.challenge','export','--output','artifacts/all_data/baseline')
Run-Python -Arguments @('-m','edgeprep.challenge','export','--output','artifacts/all_data/qdq_pointwise','--calibration','results/all_data/challenge_data/calibration.npy','--protect-ends','--protect-depthwise')
foreach ($name in @('baseline','qdq_pointwise')) {
    $suffix = if ($name -eq 'baseline') { 'fp32' } else { 'int8' }
    $model = "artifacts/all_data/$name/model_$suffix.onnx"
    Run-Python -Arguments @('-m','edgeprep.challenge','evaluate','--model',$model,'--threads','4','--output',"results/all_data/$($name)_accuracy.json")
    foreach ($threads in @(1,4)) {
        $bench = @('-m','edgeprep.benchmark','--model',$model,'--threads',"$threads",'--runs','300','--inputs','results/all_data/challenge_data/timing_input.npy','--labels','results/all_data/challenge_data/timing_label.npy','--output',"results/all_data/$($name)_t$threads.json")
        if ($threads -eq 4) { $bench += @('--energy-seconds','20') }
        Run-Python -Arguments $bench
    }
}
Run-Python -Arguments @('-m','edgeprep.local_baseline')
Run-Python -Arguments @('-m','edgeprep.challenge_report')

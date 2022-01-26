# Examples for using ML-benchmarks

## Install dependencies

    pip install transformers datasets onnx onnxruntime coloredlogs lightseq onnxruntime_tools

## Prepare ONNX model

    python -m transformers.onnx --model=bert-base-uncased .

## Prepare TorchScript model

    python export_ts.py

## Benchmark ORT model
The commoand below will create `resutls_ort.csv`:

    python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --duration 5 --batch_sizes 1 2 --sequence_lengths 10 20

## Benchmark TorchScript model
The command below will create `resutls_torchscript.csv`:

    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --duration 5 --batch_sizes 1 2 --sequence_lengths 10 20

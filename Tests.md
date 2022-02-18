# Examples for using ML-benchmarks
Transformer examples are based on https://huggingface.co/docs/transformers/serialization

## Install dependencies

    pip install transformers datasets onnx onnxruntime-gpu coloredlogs lightseq

## Prepare ONNX model

    python -m transformers.onnx --model=bert-base-uncased .

## Prepare TorchScript model

    python export_ts.py

## Benchmark Bert model, ORT mode
The commoand below will create `resutls_ort.csv`:

    python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --batch_sizes 1 2 --sequence_lengths 32
    python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --batch_sizes 1 2 --sequence_lengths 32 --gpu

## Benchmark Bert model, TorchScript mode
The command below will create `resutls_torchscript.csv`:

    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 --sequence_lengths 32
    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 --sequence_lengths 32 --gpu

## Benchmark CV model, eager mode
The command below will create `resutls_cv_eager.csv`:

    python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 --sequence_lengths 224 --gpu

## Benchmark CV model, torchscript mode
The command below will create `resutls_cv_torchscript.csv`:

    python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 --sequence_lengths 224 --gpu

## Benchmark CV model, ORT mode
The command below will create `resutls_cv_ort.csv`:

    python benchmark_runs.py --model_path cv_onnx.onnx --backend cv_ort --output_path . --batch_sizes 1 2 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_onnx.onnx --backend cv_ort --output_path . --batch_sizes 1 2 --sequence_lengths 224 --gpu

## Benchmark CV model, optimize_for_inference mode
The command below will create `resutls_cv_ofi.csv`:

    python benchmark_runs.py --model_path cv_ts.pt --backend cv_ofi --output_path . --batch_sizes 1 2 --sequence_lengths 224

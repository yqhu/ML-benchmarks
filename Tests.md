# Examples for using ML-benchmarks
Transformer examples are based on https://huggingface.co/docs/transformers/serialization

## Install dependencies

    pip install transformers datasets onnx onnxruntime-gpu coloredlogs lightseq

## Prepare ONNX model

    python -m transformers.onnx --model=bert-base-uncased .
    python -m onnxruntime.transformers.optimizer --input model.onnx --output model_fp16.onnx --float16 --opt_level 1 --model_type bert --use_gpu

## Prepare TorchScript model

    python export_ts.py

## Benchmark Bert model, ORT mode
The commoand below will create `resutls_ort.csv`:

    python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128
    python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu
    python benchmark_runs.py --model_path model_fp16.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu --fp16

## Benchmark Bert model, TorchScript mode
The command below will create `resutls_torchscript.csv`:

    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128
    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu
    python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu --fp16

## Benchmark CV model, eager mode
The command below will create `resutls_cv_eager.csv`:

    python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
    python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

## Benchmark CV model, torchscript mode
The command below will create `resutls_cv_torchscript.csv`:

    python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
    python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

## Benchmark CV model, ORT mode
The command below will create `resutls_cv_ort.csv`:

    python benchmark_runs.py --model_path cv_onnx.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224
    python benchmark_runs.py --model_path cv_onnx.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
    python benchmark_runs.py --model_path cv_onnx_fp16.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

## Benchmark CV model, optimize_for_inference mode
The command below will create `resutls_cv_ofi.csv`:

    python benchmark_runs.py --model_path cv_ts.pt --backend cv_ofi --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224

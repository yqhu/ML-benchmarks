python benchmark_runs.py --model_path bert-base-uncased --backend eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu
python benchmark_runs.py --model_path bert-base-uncased --backend eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu --fp16

python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu
python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu --fp16

python benchmark_runs.py --model_path model.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu
python benchmark_runs.py --model_path model_fp16.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --gpu --fp16

python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

python benchmark_runs.py --model_path cv_onnx.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu
python benchmark_runs.py --model_path cv_onnx_fp16.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --gpu --fp16

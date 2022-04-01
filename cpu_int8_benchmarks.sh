PREFIX=m6i.4xlarge_int8

# bert, eager, int8
python benchmark_runs.py --model_path bert-base-uncased --backend eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --num_threads 8 --int8 --prefix $PREFIX

# bert, ts, int8
python benchmark_runs.py --model_path model.pt --backend torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --num_threads 8 --int8 --prefix $PREFIX

# bert, ort, int8
python benchmark_runs.py --model_path model_int8.onnx --backend ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 128 --num_threads 8 --prefix $PREFIX

# cv, eager, int8
python benchmark_runs.py --model_path cv_eager.pt --backend cv_eager --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --num_threads 8 --int8 --prefix $PREFIX

# cv, ts, int8
python benchmark_runs.py --model_path cv_ts.pt --backend cv_torchscript --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --num_threads 8 --int8 --prefix $PREFIX

# cv, ort, int8
python benchmark_runs.py --model_path cv_onnx_int8.onnx --backend cv_ort --output_path . --batch_sizes 1 2 4 8 12 16 32 64 --sequence_lengths 224 --num_threads 8 --prefix $PREFIX


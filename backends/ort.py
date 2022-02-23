from pathlib import Path
from transformers.convert_graph_to_onnx import convert
# from onnxruntime_tools import optimizer
# from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
from transformers import BertTokenizerFast
import onnxruntime

from os import environ
from psutil import cpu_count
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from transformers import BertTokenizerFast, DistilBertTokenizer
from onnxruntime.transformers import optimizer
import onnx
import numpy as np
from time import perf_counter
import torch
from transformers import TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer, SEC_TO_MS_SCALE
import csv 
import multiprocessing as mp


def benchmark_ORT(model_path, batch_size,sequence_length, backend, output_folder, duration=1000, num_threads=-1, gpu=False, fp16=False):
    if num_threads < 0:
        num_threads = mp.cpu_count()
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads

    
    if gpu and torch.cuda.is_available():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    model = onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=providers)   

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.NUMPY,
    )
    inputs = {k: v.astype("i8") for k, v in inputs.items()}
    # inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    latencies = []
    # Warmup
    for _ in range(10):
        _ = model.run(None, inputs)
    
    for _ in range(duration):
        start_time = perf_counter()
        _ = model.run(None, inputs)
        latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
        latencies.append(latency)
    
    # Compute run statistics
    print(f"******* batch_size = {batch_size}, sequence_length = {sequence_length}, {sum(latencies)} ms / {duration} iters")
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(duration * batch_size * SEC_TO_MS_SCALE / np.sum(latencies), 2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
    

def profile_ORT(model_path, batch_size,sequence_length, output_folder):
    model = onnxruntime.InferenceSession(model_path)   

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.NUMPY,
    )
    inputs = {k: v.astype("i8") for k, v in inputs.items()}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
    wait=2,
    warmup=2,
    active=6,
    repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/{}'.format(output_folder)),
    with_stack=True) as profiler:
        for i in range(1000):
            _ = model.run(None, inputs)
            profiler.step()


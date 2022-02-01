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
import torchvision
from transformers import TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer, SEC_TO_MS_SCALE
import csv 
import multiprocessing as mp


def benchmark_Eager(model_path, batch_size, sequence_length, backend, output_folder, duration, num_threads=-1):
    if num_threads < 0:
        num_threads = mp.cpu_count()

    torch.set_num_threads(num_threads)

    model = torch.load(model_path)
    model.eval()

    inputs = torch.rand((batch_size, 3, sequence_length, sequence_length))

    latencies = []

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            # _ = model.run(None, inputs)
            _ = model(inputs)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    with torch.inference_mode():
        while sum(latencies) < duration:
            start_time = perf_counter()
            # _ = model.run(None, inputs)
            _ = model(inputs)
            latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
            latencies.append(latency)
        
    # Compute run statistics
    print("*******", len(latencies), sum(latencies))
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(((len(latencies)/duration)*batch_size)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
    

def benchmark_TorchScript(model_path, batch_size, sequence_length, backend, output_folder, duration, num_threads=-1):
    if num_threads < 0:
        num_threads = mp.cpu_count()

    torch.set_num_threads(num_threads)

    model = torch.jit.load(model_path)
    model.eval()

    inputs = torch.rand((batch_size, 3, sequence_length, sequence_length))

    latencies = []

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            # _ = model.run(None, inputs)
            _ = model(inputs)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    with torch.inference_mode():
        while sum(latencies) < duration:
            start_time = perf_counter()
            # _ = model.run(None, inputs)
            _ = model(inputs)
            latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
            latencies.append(latency)
        
    # Compute run statistics
    print("*******", len(latencies), sum(latencies))
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(((len(latencies)/duration)*batch_size)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
    
def benchmark_OFI(model_path, batch_size, sequence_length, backend, output_folder, duration, num_threads=-1):
    if num_threads < 0:
        num_threads = mp.cpu_count()

    torch.set_num_threads(num_threads)

    model = torch.jit.load(model_path)
    model = torch.jit.optimize_for_inference(model.eval())

    inputs = torch.rand((batch_size, 3, sequence_length, sequence_length))

    latencies = []

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            # _ = model.run(None, inputs)
            _ = model(inputs)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    with torch.inference_mode():
        while sum(latencies) < duration:
            start_time = perf_counter()
            # _ = model.run(None, inputs)
            _ = model(inputs)
            latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
            latencies.append(latency)
        
    # Compute run statistics
    print("*******", len(latencies), sum(latencies))
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(((len(latencies)/duration)*batch_size)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
    
def benchmark_CV_ORT(model_path, batch_size, sequence_length, backend, output_folder, duration, num_threads=-1):
    if num_threads < 0:
        num_threads = mp.cpu_count()

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads

    model = onnxruntime.InferenceSession(model_path, sess_options=sess_options)   

    inputs = torch.rand((batch_size, 3, sequence_length, sequence_length))
    inputs = {'input': inputs.numpy()}

    latencies = []

    # Warmup
    for _ in range(10):
        _ = model.run(None, inputs)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    with torch.inference_mode():
        while sum(latencies) < duration:
            start_time = perf_counter()
            _ = model.run(None, inputs)
            latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
            latencies.append(latency)
        
    # Compute run statistics
    print("*******", len(latencies), sum(latencies))
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(((len(latencies)/duration)*batch_size)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
from pathlib import Path
from os import environ
from psutil import cpu_count
from transformers import BertTokenizerFast, DistilBertTokenizer
import numpy as np
from time import perf_counter
import torch
from backends.ort import benchmark_ORT
from transformers import AutoModel, TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer, SEC_TO_MS_SCALE
import csv

def benchmark_Torchscript(model_path, batch_size,sequence_length, backend, output_folder, duration, num_threads=-1, gpu=False):
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    device = torch.device("cuda" if gpu else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    model = torch.jit.load(model_path, map_location=device)
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.PYTORCH,
    )
    model_inputs = inputs
    latencies = []
    # Warmup
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device) 
    for _ in range(10):
        _ = model(input_ids,attention_mask)
    for _ in range(duration):
        start_time = perf_counter()
        _ = model(input_ids,attention_mask)
        latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
        latencies.append(latency)

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
    
def profile_torchscript(model_path, batch_size,sequence_length, output_folder, gpu=False):
    device = torch.device("cuda" if gpu else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    model = torch.jit.load(model_path, map_location=device)
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.PYTORCH,
    )
    model_inputs = inputs
    latencies = []
    # Warmup
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device) 

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
    wait=2,
    warmup=2,
    active=6,
    repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/{}'.format(output_folder)),
    with_stack=True) as profiler:
        for i in range(1000):
            _ = model(input_ids,attention_mask)
            profiler.step()
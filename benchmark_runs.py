from backends.ort import benchmark_ORT, profile_ORT
from backends.torchscript import benchmark_Torchscript, profile_torchscript
from backends.lightseq import benchmark_LightSeq, profile_LightSeq
from backends.cv import benchmark_Eager, benchmark_TorchScript, benchmark_OFI, benchmark_CV_ORT
from utils.utils import csv_writer
import os
import multiprocessing as mp
import pandas as pd

from argparse import ArgumentParser

def run_worker(args):
    benchmarks_list =[]
    for batch_size in args.batch_sizes:
        for sequence_length in args.sequence_lengths:
            if args.backend == 'ort':
                benchmarks_list.append(benchmark_ORT(args.model_path, batch_size,sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
                #csv_writer(benchmarks_list, args.backend, args.output_path)
            elif args.backend == 'torchscript':
                benchmarks_list.append(benchmark_Torchscript(args.model_path, batch_size,sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
                #csv_writer(benchmarks_list, args.backend, args.output_path)
            elif args.backend == 'cv_eager':
                benchmarks_list.append(benchmark_Eager(args.model_path, batch_size, sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
            elif args.backend == 'cv_torchscript':
                benchmarks_list.append(benchmark_TorchScript(args.model_path, batch_size, sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
            elif args.backend == 'cv_ofi':
                benchmarks_list.append(benchmark_OFI(args.model_path, batch_size, sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
            elif args.backend == 'cv_ort':
                benchmarks_list.append(benchmark_CV_ORT(args.model_path, batch_size, sequence_length, args.backend, args.output_path, args.duration, num_threads=args.num_threads, gpu=args.gpu))
            else:
                pass

    return benchmarks_list


if __name__ == '__main__':
    parser = ArgumentParser("Model benchmarking")
    parser.add_argument("--model_path", type=str, help="The path to the trained/optimzied model")
    parser.add_argument("--duration", type=int, default=1000, help="cycles of benchmark run")
    parser.add_argument("--backend", type=str, help="Backend, torchscript, ort, or lightseq")
    parser.add_argument("--output_path", type=str, help="Where the resulting report will be saved")
    parser.add_argument("--profile", type=bool, help="flag to profile the model")
    parser.add_argument('--batch_sizes', nargs='+', type=int)
    parser.add_argument('--sequence_lengths', nargs='+', type=int)
    parser.add_argument('--num_workers', default=1, type=int, help='The number of concurrent workers. Default: 1')
    parser.add_argument('--num_threads', default=-1, type=int, help='The number of threads per worker. Default: -1 (no limitation)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU. Default: Use CPU')
    # Parse command line arguments
    args = parser.parse_args()

    if args.profile and args.backend=='ort':
        profile_ORT(args.model_path, args.batch_sizes[0],args.sequence_lengths[0], args.output_path)
    elif args.profile and args.backend=='torchscript':
        profile_torchscript(args.model_path, args.batch_sizes[0],args.sequence_lengths[0], args.output_path)
    elif args.profile and args.backend=='lightseq':
        profile_LightSeq(args.model_path, args.batch_sizes[0],args.sequence_lengths[0], args.output_path)
    else:
        args_repeats = [args] * args.num_workers
        with mp.Pool(args.num_workers) as pool:
            ret = pool.map(run_worker, args_repeats)

        df = pd.DataFrame(ret[0])
        cols = df.columns[2:]
        for r in ret[1:]:
            df[cols] += pd.DataFrame(r)[cols]

        cols = list(cols)
        cols.remove('throughput')
        df[cols] /= len(ret)

        gpu = '_gpu' if args.gpu else ''
        file_name = os.path.join(args.output_path, f"resutls_{args.backend}{gpu}.csv")
        df.to_csv(file_name, index=False)

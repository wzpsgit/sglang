import os
import time
import logging
import json
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from sglang.srt.managers.io_struct import ProfileReq

logger = logging.getLogger(__name__)

class FunctionProfiler:
    """
    Store MX global profiling settings.
    """

    def __init__(self):
        self.profile_funcs = None
        self.profiler_activities = [ProfilerActivity.CUDA]        
        self.record_shapes = False
        self.profile_memory = False
        self.with_stack = True
        self.trace_dir = None       
        self.profile_steps = []
        self.step_counters = {}
        self.round_counter = 0
        self.tp_ranks = [1]

        self.skip_first = 0
        self.wait = 0
        self.active = 1        
        self.repeat = 0
        
    def start_profile(self, 
                    tp_rank,
                    req: ProfileReq 
    ):
        self.activities = req.activities
        if(self.activities is None):
            self.activities = ["CPU", "GPU"]
        self.tp_ranks = req.tp_ranks if req.tp_ranks is not None else [1]
        if(tp_rank not in self.tp_ranks):
            return 

        self.profile_funcs = req.profile_funcs
        if(len(req.profile_funcs) == 0):
            return  

        self.round_counter += 1
        self.step_counters = {} 
        self.profile_steps = req.profile_steps if req.profile_steps is not None else [] 
        self.trace_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp") 
        if(req.output_dir is None):
            # create trace directory
            current_time = time.time()
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(current_time))   
            self.trace_dir = os.path.join(self.trace_dir, f"trace{self.round_counter}_{timestamp}")
        else:
            self.trace_dir = os.path.join(self.trace_dir, req.output_dir)
            folder_name = "_".join(self.profile_funcs) + f"-R{self.round_counter}-S_" \
                        + "_".join([str(s) for s in self.profile_steps])
            self.trace_dir = os.path.join(self.trace_dir, folder_name)
        os.makedirs(self.trace_dir, exist_ok=True) 

        self.profiler_activities = [ProfilerActivity.CUDA]
        if(self.activities is not None):
            activity_map = {
                "CPU": ProfilerActivity.CPU,
                "GPU": ProfilerActivity.CUDA,
            }
            self.profiler_activities = [
                activity_map[a] for a in self.activities if a in activity_map
            ]  
        
        self.with_stack = req.with_stack if req.with_stack is not None else True
        self.record_shapes = req.record_shapes if req.record_shapes is not None else False
        self.profile_memory = req.profile_memory if req.profile_memory is not None else False
        self.skip_first = req.skip_first if req.skip_first is not None else 0
        self.wait = req.wait if req.wait is not None else 0
        self.active = req.active if req.active is not None else 1
        self.repeat = req.repeat if req.repeat is not None else 0       

        logger.info(f"MX profiler starts: trace_dir={self.trace_dir},\n \
            tp_ranks={self.tp_ranks}, activities={self.activities}, \
            record_shapes={self.record_shapes}, profile_memory={self.profile_memory}, \
            with_stack={self.with_stack}, profile_funcs={self.profile_funcs}, \
            profile_steps={self.profile_steps}, skip_first={self.skip_first}, wait={self.wait}, \
            active={self.active}, repeat={self.repeat}")   

    def stop_profile(self, tp_rank):
        if(self.profile_funcs is None or tp_rank not in self.tp_ranks):
            return

        logger.info(f"MX profiler stops. Function step counter stat: {self.step_counters}")

        # Append results to a JSONL file
        output_file_name = os.path.join(self.trace_dir, "func_steps.jsonl")
        with open(output_file_name, "a") as file:
            file.write(json.dumps(self.step_counters) + "\n")        
        self.profile_funcs = None

        if(tp_rank not in self.tp_ranks):
            return               
    
        self.step_counters = {} 
        
    def should_profile(self, func_name):
        return self.profile_funcs is not None \
                and func_name in self.profile_funcs

    def export_profiler_to_csv(self, profiler, output_file, profile_mem=False):    
        df = pd.DataFrame(columns=[
            'key', 'self_cpu', 'self_cpu_avg','cpu_total', 'cpu_total_avg',
            'self_gpu', 'self_gpu_avg', 'gpu_total', 'gpu_total_avg',
            'cpu_mem', 'self_cpu_mem', 'num_calls'
        ])

        for event in profiler.key_averages():
            df.loc[len(df)] = [
                event.key,
                f"{event.self_cpu_time_total / 1000:.4f}",
                f"{event.self_cpu_time_total / event.count / 1000:.4f}",
                f"{event.cpu_time_total / 1000:.4f}",
                f"{event.cpu_time_total / event.count / 1000:.4f}",
                f"{event.self_device_time_total / 1000:.4f}",
                f"{event.self_device_time_total / event.count / 1000:.4f}",                
                f"{event.device_time_total / 1000:.4f}",
                f"{event.device_time_total / event.count / 1000:.4f}",
                f"{event.cpu_memory_usage / 1024 / 1024:.4f}",
                f"{event.self_cpu_memory_usage / 1024 / 1024:.4f}",
                event.count
            ]
        
        df.to_csv(output_file, index=False)

    def run(self, tp_rank, name, func, *args):
        if(not self.should_profile(name)):
            return func(*args)

        if(tp_rank not in self.tp_ranks):
            return func(*args)

        if name in self.step_counters:
            self.step_counters[name] += 1
        else:
            self.step_counters[name] = 1

        if(len(self.profile_steps) == 0):        
            # skip first N steps
            if(self.step_counters[name] <= self.skip_first):
                logger.debug(f"Skip profiling function={name}, step={self.step_counters[name]}, skip_first={self.skip_first}")
                return func(*args)
            
            # skip profiling if all profiling rounds completed
            cur_round = (self.step_counters[name] - self.skip_first) // (self.wait + self.active)
            if(cur_round > self.repeat):
                logger.debug(f"Skip profiling function={name}, step={self.step_counters[name]}, cur_round={cur_round}")
                return func(*args)  

            # skip wait steps
            round_step = (self.step_counters[name] - self.skip_first + self.wait - 1) % (self.wait + self.active)
            if (round_step < self.wait):
                logger.debug(f"Skip profiling function={name}, step={self.step_counters[name]}, round_step={round_step}")
                return func(*args) 
        else: 
            if(self.step_counters[name] not in self.profile_steps):
                logger.debug(f"Skip profiling function={name}, step={self.step_counters[name]}, profile_steps={self.profile_steps}")
                return func(*args)
 
        logger.info(f"Profiling function={name}, activities={self.activities}, step={self.step_counters[name]}") 
        #cuda profiling
        if(self.activities is not None):
            if("MEM" in self.activities):
                torch.cuda.memory._record_memory_history(max_entries=100000)
                result = func(*args)
                memory_profile_path = os.path.join(
                    self.trace_dir,
                    f"{name}-TP-{tp_rank}-memory" + str(time.time()) + ".pickle",
                )
                logger.info(f"Outputting memory snapshot {memory_profile_path}")
                torch.cuda.memory._dump_snapshot(memory_profile_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                return result
            elif("CUDA_PROFILER" in self.activities):
                torch.cuda.cudart().cudaProfilerStart()
                result = func(*args)
                torch.cuda.cudart().cudaProfilerStop()
                return result

        def trace_handler(p):
            trace_csv = f"{self.trace_dir}/{name}-TP{tp_rank}-{self.step_counters[name]}.csv"
            trace_file = f"{self.trace_dir}/{name}-TP{tp_rank}-{self.step_counters[name]}.json"
            logger.info(f"Outputting trace files: {trace_csv}, {trace_file}")
            #Outputting csv file
            self.export_profiler_to_csv(p, trace_csv, self.profile_memory)
            #Outputting trace file
            p.export_chrome_trace(trace_file)

        with profile(
            activities=self.profiler_activities,
            on_trace_ready=trace_handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            with record_function(name):
                result = func(*args)

        return result  

profiler = FunctionProfiler()
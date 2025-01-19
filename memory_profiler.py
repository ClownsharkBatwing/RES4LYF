import torch
import gc
import time
import logging
from pytorch_memlab import MemReporter
from comfy.model_management import module_size
from typing import Dict, Set, Tuple
import sys

def print_tensors(threshold_mb=10, sort_by_size=False):
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.numel() / (1024 * 1024)
                ref_count = sys.getrefcount(obj) - 3  # Subtract 3 for getrefcount's own references
                if size_mb > threshold_mb:
                    tensors.append((size_mb, obj.shape, ref_count))
        except Exception:
            pass
    
    if sort_by_size:
        tensors.sort(reverse=True, key=lambda x: x[0])
    
    for size_mb, shape, ref_count in tensors:
        print(f"Size: {size_mb:.2f}MB | Shape: {shape} | RefCount: {ref_count}")

def get_tensor_size(tensor):
    return tensor.element_size() * tensor.numel() / (1024 * 1024)  # in MB

def get_module_size(module):
    return module_size(module) / (1024 * 1024)  # in MB

class MemoryTracker:
    def __init__(self, model=None, threshold_mb=100):
        self.reporter = MemReporter(model)
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.sampling_points = {}
        self.tensor_snapshots = {}  # Store tensor info at each checkpoint
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MemoryTracker")
    
    def _get_tensor_key(self, shape, dtype) -> str:
        """Create a unique key for tensor shape/type combination"""
        return f"{shape}_{dtype}"
    
    def _capture_tensors(self) -> Dict[str, Tuple[float, torch.Size, int]]:
        """Capture all CUDA tensors above threshold"""
        tensors = {}
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_mb = obj.element_size() * obj.numel() / (1024 * 1024)
                    if size_mb > self.threshold_bytes / (1024 * 1024):
                        key = self._get_tensor_key(obj.shape, obj.dtype)
                        ref_count = sys.getrefcount(obj) - 3
                        tensors[key] = (size_mb, obj.shape, obj.dtype, ref_count)
            except Exception:
                pass
        return tensors

    def checkpoint(self, name, verbose=False):
        """Take a memory snapshot at a specific point"""
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        self.sampling_points[name] = {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'timestamp': time.time()
        }
        
        # Capture tensor snapshot
        self.tensor_snapshots[name] = self._capture_tensors()
        
        self.logger.info(f"\n=== Memory Checkpoint: {name} ===")
        self.logger.info(f"Allocated: {allocated:.2f}MB")
        self.logger.info(f"Reserved: {reserved:.2f}MB")
        self.logger.info(f"Peak: {max_allocated:.2f}MB")
        
        if verbose:
            self.logger.info("\nCurrent tensors in memory:")
            for _, (size, shape, dtype, ref_count) in sorted(
                self.tensor_snapshots[name].items(), 
                key=lambda x: x[1][0], 
                reverse=True
            ):
                self.logger.info(
                    f"Size: {size:.2f}MB | Shape: {shape} | "
                    f"Type: {dtype} | RefCount: {ref_count}"
                )

    def compare_points(self, point1, point2):
        """Compare memory usage between two checkpoints"""
        if point1 not in self.sampling_points or point2 not in self.sampling_points:
            self.logger.error("Invalid checkpoint names")
            return
            
        p1 = self.sampling_points[point1]
        p2 = self.sampling_points[point2]
        
        diff_allocated = p2['allocated'] - p1['allocated']
        self.logger.info(f"\nMemory change between {point1} and {point2}:")
        self.logger.info(f"Difference in allocated memory: {diff_allocated:.2f}MB")
        
        # Compare tensor snapshots
        tensors1 = set(self.tensor_snapshots[point1].keys())
        tensors2 = set(self.tensor_snapshots[point2].keys())
        
        # Find new tensors
        new_tensors = tensors2 - tensors1
        if new_tensors:
            self.logger.info("\nNew tensors:")
            for key in new_tensors:
                size, shape, dtype, ref_count = self.tensor_snapshots[point2][key]
                self.logger.info(
                    f"Size: {size:.2f}MB | Shape: {shape} | "
                    f"Type: {dtype} | RefCount: {ref_count}"
                )
        
        # Find removed tensors
        removed_tensors = tensors1 - tensors2
        if removed_tensors:
            self.logger.info("\nRemoved tensors:")
            for key in removed_tensors:
                size, shape, dtype, ref_count = self.tensor_snapshots[point1][key]
                self.logger.info(
                    f"Size: {size:.2f}MB | Shape: {shape} | "
                    f"Type: {dtype} | RefCount: {ref_count}"
                )

def auto_track_model_accumulation(tracker: MemoryTracker, checkpoint_name: str, step: int, sub_step: int = None, auto_save: bool = True, verbose: bool = True):
    """Helper function to track memory at specific points and optionally save to file"""
    import os
    import datetime
    
    # Create checkpoint name based on parameters
    name = f"{checkpoint_name}_{step}" if sub_step is None else f"{checkpoint_name}_{step}_{sub_step}"
    
    # Take memory snapshot
    tracker.checkpoint(name, verbose=verbose)
    
    if auto_save:
        # Create logs directory if it doesn't exist
        os.makedirs("memory_logs", exist_ok=True)
        
        # Generate filename with timestamp to avoid overwrites
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("memory_logs", f"{name}_{timestamp}.txt")
        
        # Save checkpoint data to file
        with open(filename, "w") as f:
            f.write(f"=== Memory Checkpoint: {name} ===\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            checkpoint_data = tracker.sampling_points[name]
            f.write(f"Allocated: {checkpoint_data['allocated']:.2f}MB\n")
            f.write(f"Reserved: {checkpoint_data['reserved']:.2f}MB\n")
            f.write(f"Peak: {checkpoint_data['max_allocated']:.2f}MB\n\n")
            
            f.write("Current tensors in memory:\n")
            for key, (size, shape, dtype, ref_count) in sorted(
                tracker.tensor_snapshots[name].items(),
                key=lambda x: x[1][0],
                reverse=True
            ):
                f.write(f"Size: {size:.2f}MB | Shape: {shape} | Type: {dtype} | RefCount: {ref_count}\n")

class AutoMemoryTracker(MemoryTracker):
    def __init__(self, model=None, threshold_mb=100, auto_save=True, verbose=True):
        super().__init__(model, threshold_mb)
        self.auto_save = auto_save
        self.verbose = verbose
        self.tracking_enabled = True
        
    def track_step(self, checkpoint_name: str, step: int, sub_step: int = None):
        """Convenience method to track memory at a specific step"""
        if self.tracking_enabled:
            auto_track_model_accumulation(
                self,
                checkpoint_name,
                step,
                sub_step,
                self.auto_save,
                self.verbose
            )
    
    def start_tracking(self):
        """Enable memory tracking"""
        self.tracking_enabled = True
    
    def stop_tracking(self):
        """Disable memory tracking"""
        self.tracking_enabled = False
    
    def get_checkpoint_filenames(self):
        """Get list of all checkpoint files in the memory_logs directory"""
        import glob
        import os
        return sorted(glob.glob(os.path.join("memory_logs", "*.txt")))

    def analyze_accumulation(self, steps_range=None, size_threshold_mb=10):
        """
        Analyze memory accumulation patterns across checkpoints.
        
        Args:
            steps_range: Optional tuple of (start_step, end_step) to analyze
            size_threshold_mb: Only track tensors larger than this size (in MB)
        """
        checkpoints = sorted(self.sampling_points.items(), key=lambda x: x[1]['timestamp'])
        
        if steps_range:
            start, end = steps_range
            filtered_checkpoints = []
            for cp in checkpoints:
                # Parse checkpoint name to handle both "name_step" and "name_step_substep" formats
                parts = cp[0].split('_')
                if len(parts) >= 2:
                    try:
                        step_num = int(parts[1])
                        if start <= step_num <= end:
                            filtered_checkpoints.append(cp)
                    except ValueError:
                        continue
            checkpoints = filtered_checkpoints

        analysis = {
            'total_memory_trend': [],
            'tensor_counts': [],
            'persistent_tensors': set(),
            'accumulated_tensors': {},
            'memory_spikes': []
        }
        
        prev_tensors = None
        baseline_memory = None
        
        for i, (checkpoint_name, checkpoint_data) in enumerate(checkpoints):
            current_tensors = self.tensor_snapshots[checkpoint_name]
            total_memory = checkpoint_data['allocated']
            
            # Track total memory trend
            analysis['total_memory_trend'].append({
                'checkpoint': checkpoint_name,
                'total_memory': total_memory,
                'reserved_memory': checkpoint_data['reserved']
            })
            
            # Track tensor counts and sizes
            analysis['tensor_counts'].append({
                'checkpoint': checkpoint_name,
                'count': len(current_tensors),
                'total_size': sum(size for size, _, _, _ in current_tensors.values())
            })

            # Identify memory spikes
            if baseline_memory is None:
                baseline_memory = total_memory
            elif total_memory > baseline_memory * 1.5:  # 50% increase threshold
                analysis['memory_spikes'].append({
                    'checkpoint': checkpoint_name,
                    'memory_increase': total_memory - baseline_memory,
                    'memory': total_memory,
                    'baseline': baseline_memory
                })
            
            if prev_tensors is not None:
                # Find new tensors
                new_tensors = set(current_tensors.keys()) - set(prev_tensors.keys())
                removed_tensors = set(prev_tensors.keys()) - set(current_tensors.keys())
                
                # Track persistent tensors
                if i == 1:
                    analysis['persistent_tensors'] = set(current_tensors.keys())
                else:
                    analysis['persistent_tensors'] &= set(current_tensors.keys())
                
                # Track accumulated tensors
                for tensor_key in new_tensors:
                    size, shape, dtype, ref_count = current_tensors[tensor_key]
                    if size >= size_threshold_mb:
                        if tensor_key not in analysis['accumulated_tensors']:
                            analysis['accumulated_tensors'][tensor_key] = {
                                'first_seen': checkpoint_name,
                                'size': size,
                                'shape': shape,
                                'dtype': dtype,
                                'initial_ref_count': ref_count
                            }
                
                # Log significant changes
                self.logger.info(f"\n=== Changes at {checkpoint_name} ===")
                
                if new_tensors:
                    self.logger.info(f"\nNew tensors (>{size_threshold_mb}MB):")
                    new_memory = 0
                    for key in new_tensors:
                        size, shape, dtype, ref_count = current_tensors[key]
                        if size >= size_threshold_mb:
                            self.logger.info(f"+ Size: {size:.2f}MB | Shape: {shape} | Type: {dtype} | RefCount: {ref_count}")
                            new_memory += size
                    self.logger.info(f"Total new memory: {new_memory:.2f}MB")
                
                if removed_tensors:
                    self.logger.info(f"\nRemoved tensors (>{size_threshold_mb}MB):")
                    freed_memory = 0
                    for key in removed_tensors:
                        size, shape, dtype, ref_count = prev_tensors[key]
                        if size >= size_threshold_mb:
                            self.logger.info(f"- Size: {size:.2f}MB | Shape: {shape} | Type: {dtype} | RefCount: {ref_count}")
                            freed_memory += size
                    self.logger.info(f"Total freed memory: {freed_memory:.2f}MB")
            
            prev_tensors = current_tensors

        # Print final analysis summary
        self._print_analysis_summary(analysis, size_threshold_mb, checkpoints[-1][1] if checkpoints else None)
        
        return analysis

    def _print_analysis_summary(self, analysis, size_threshold_mb, final_checkpoint_data):
        """Helper method to print the analysis summary"""
        self.logger.info("\n=== Memory Analysis Summary ===")
        
        # Memory trend
        memory_trend = analysis['total_memory_trend']
        if memory_trend:
            self.logger.info(f"\nMemory Trend:")
            self.logger.info(f"Initial Memory: {memory_trend[0]['total_memory']:.2f}MB")
            self.logger.info(f"Final Memory: {memory_trend[-1]['total_memory']:.2f}MB")
            self.logger.info(f"Peak Memory: {max(x['total_memory'] for x in memory_trend):.2f}MB")
        
        # Persistent tensors
        if final_checkpoint_data is not None:
            self.logger.info(f"\nPersistent Tensors (present throughout all steps):")
            persistent_memory = 0
            current_tensors = self.tensor_snapshots[list(final_checkpoint_data.keys())[0]]
            for key in analysis['persistent_tensors']:
                size, shape, dtype, ref_count = current_tensors[key]
                if size >= size_threshold_mb:
                    self.logger.info(f"Size: {size:.2f}MB | Shape: {shape} | Type: {dtype} | RefCount: {ref_count}")
                    persistent_memory += size
            self.logger.info(f"Total persistent memory: {persistent_memory:.2f}MB")
        
        # Memory spikes
        if analysis['memory_spikes']:
            self.logger.info(f"\nSignificant Memory Spikes:")
            for spike in analysis['memory_spikes']:
                self.logger.info(f"Checkpoint: {spike['checkpoint']}")
                self.logger.info(f"Memory at spike: {spike['memory']:.2f}MB")
                self.logger.info(f"Baseline: {spike['baseline']:.2f}MB")
                self.logger.info(f"Increase: {spike['memory_increase']:.2f}MB")
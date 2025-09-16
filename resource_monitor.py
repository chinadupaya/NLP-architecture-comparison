import threading
import psutil
import GPUtil
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import platform
import subprocess

# TO DO export to csv for comparison between architectures

def get_is_mac():
    if (platform.system() == 'Darwin'):
        return True
    return False

class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        self.is_mac = False

    def start_monitoring(self, interval=1.0):
        self.monitoring=True
        self.monitor_thread = threading.Thread(target=self.monitor_main, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        # check if mac device
        self.is_mac = get_is_mac()
        print("Resource monitoring started...")

    def stop_monitoring(self):
        # reset everything
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Resource monitoring stopped.")

    def monitor_main(self, interval):
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory.percent)
            self.timestamps.append(datetime.now())
            
            # GPU monitoring (if available)
            if self.is_mac:
                print("mac thingz")
                # mac
            else:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Monitor first GPU
                        self.gpu_usage.append(gpu.load * 100)
                        self.gpu_memory.append(gpu.memoryUtil * 100)
                    else:
                        self.gpu_usage.append(0)
                        self.gpu_memory.append(0)
                except:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
            time.sleep(interval)

    def plot_all(self):
        fig, axes = plt.subplots(2,2, figsize=(15,10))
        fig.suptitle('Resource utilization during training')

        start_time = self.timestamps[0]
        # convert all timestamps to relative time
        time_minutes = [(t - start_time).total_seconds() / 60 for t in self.timestamps]

        # cpu
        axes[0, 0].plot(time_minutes, self.cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)

        # mem
        axes[0, 1].plot(time_minutes, self.memory_usage, 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)

        # gpu
        axes[1, 0].plot(time_minutes, self.gpu_usage, 'r-', linewidth=2)
        axes[1, 0].set_title('GPU Usage (%)')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('GPU %')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)

        # gpu mem
        axes[1, 1].plot(time_minutes, self.gpu_memory, 'orange', linewidth=2)
        axes[1, 1].set_title('GPU Memory Usage (%)')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('GPU Memory %')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()

    def get_summary_stats(self):
        stats = {
            'CPU': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
                'std': np.std(self.cpu_usage)
            },
            'Memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
                'std': np.std(self.memory_usage)
            },
            'GPU': {
                'mean': np.mean(self.gpu_usage),
                'max': np.max(self.gpu_usage),
                'min': np.min(self.gpu_usage),
                'std': np.std(self.gpu_usage)
            },
            'GPU_Memory': {
                'mean': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory),
                'min': np.min(self.gpu_memory),
                'std': np.std(self.gpu_memory)
            }
        }

        print("\n=== Resource Usage Summary ===")
        for resource, values in stats.items():
            print(f"\n{resource}:")
            print(f"  Mean: {values['mean']:.1f}%")
            print(f"  Max:  {values['max']:.1f}%")
            print(f"  Min:  {values['min']:.1f}%")
            print(f"  Std:  {values['std']:.1f}%")

class ResourceMonitorCB(tf.keras.callbacks.Callback):
    def __init__(self, monitor_interval=5.0):
        super().__init__()
        self.monitor = ResourceMonitor()
        self.monitor_interval = monitor_interval
        
    def on_train_begin(self, logs=None):
        print("Starting resource monitoring...")
        self.monitor.start_monitoring(self.monitor_interval)
        
    def on_train_end(self, logs=None):
        print("Stopping resource monitoring...")
        self.monitor.stop_monitoring()
        
    def on_epoch_end(self, epoch, logs=None):
        # Print current resource usage every epoch
        if self.monitor.cpu_usage:
            current_cpu = self.monitor.cpu_usage[-1]
            current_memory = self.monitor.memory_usage[-1]
            current_gpu = self.monitor.gpu_usage[-1] if self.monitor.gpu_usage else 0
            print(f"Epoch {epoch+1} - CPU: {current_cpu:.1f}%, Memory: {current_memory:.1f}%, GPU: {current_gpu:.1f}%")

# Function to get system information
def print_system_info():
    print("=== System Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # CPU info
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # GPU info
    if get_is_mac():
        has_metal = False
        try:
            result = subprocess.run([
                'system_profiler', 'SPDisplaysDataType'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout
                # Look for Metal support (indicates Apple Silicon or dedicated GPU)
                if "Metal" in output:
                    has_metal = True
            has_metal = False
        except:
            has_metal = False
        print(f"Metal GPU Support: {'Yes' if has_metal else 'No'}")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"GPUs detected by TensorFlow: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
        # Additional GPU info using GPUtil
        try:
            gpu_list = GPUtil.getGPUs()
            for i, gpu in enumerate(gpu_list):
                print(f"  GPU {i} Details: {gpu.name}, Memory: {gpu.memoryTotal}MB")
            # Check for MPS availability on Mac
            if get_is_mac() and hasattr(tf.config.experimental, 'list_physical_devices'):
                try:
                    mps_devices = tf.config.experimental.list_physical_devices('GPU')
                    if mps_devices:
                        print(f"  MPS (Metal Performance Shaders) available: Yes")
                    else:
                        print(f"  MPS (Metal Performance Shaders) available: No")
                except:
                    pass
        except:
            pass
            
    except:
        print("No GPUs detected")
    
    print("=" * 20)

# # Function to check TensorFlow GPU usage
# def print_tensorflow_gpu_info():
#     """Print detailed system information."""
#     print("\n=== TensorFlow GPU Configuration ===")
    
#     # Check if GPU is available
#     print(f"GPU Available: {tf.config.experimental.list_physical_devices('GPU')}")
#     print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
#     # Memory growth configuration (recommended)
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             print("GPU memory growth enabled")
#         except RuntimeError as e:
#             print(f"GPU configuration error: {e}")
    
#     print("=" * 20)
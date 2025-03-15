import time
import os
import psutil
import threading
import tiktoken

from .cost_calculator import CostCalculator
from .logger import Logger
from .utils import calculate_carbon_emissions
from .visualizer import Visualizer

class AgentMont:
    def __init__(self, model: str, enable_monitoring: bool = True, encoding_name: str = "cl100k_base"):
        self.model = model
        self.start_time = None
        self.end_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0
        self.cpu_usage = []
        self.memory_usage = []
        self.latencies = []  # List to record individual operation latencies
        self.logger = Logger()
        self.cost_calculator = CostCalculator(model)
        self.carbon_emissions = 0.0
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.enable_monitoring = enable_monitoring
        self.monitoring = False
        self.monitor_thread = None

    def start(self):
        self.logger.log("AgentMont monitoring started.")
        self.start_time = time.time()
        if self.enable_monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.start()

    def _monitor_resources(self):
        process = psutil.Process(os.getpid())
        while self.monitoring:
            cpu = process.cpu_percent(interval=1)
            mem = process.memory_info().rss / (1024 * 1024)  # MB
            self.cpu_usage.append(cpu)
            self.memory_usage.append(mem)
            time.sleep(1)

    def record_latency(self, latency: float):
        self.latencies.append(latency)
        self.logger.log(f"Recorded latency: {latency:.4f} seconds")

    def measure_operation(self, func, *args, **kwargs):
        """
        Wraps a callable to measure its execution latency.
        """
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        self.record_latency(latency)
        return result

    def set_token_counts(self, input_text: str, output_text: str):
        self.input_tokens = self.count_tokens(input_text)
        self.output_tokens = self.count_tokens(output_text)
        self.total_tokens = self.input_tokens + self.output_tokens
        self.logger.log(f"Input Tokens: {self.input_tokens}")
        self.logger.log(f"Output Tokens: {self.output_tokens}")
        self.logger.log(f"Total Tokens: {self.total_tokens}")

    def set_token_usage_from_crew_output(self, crew_output):
        """
        Extracts token usage from a CrewOutput object.
        """
        if hasattr(crew_output, 'token_usage') and crew_output.token_usage:
            self.input_tokens = crew_output.token_usage.prompt_tokens
            self.output_tokens = crew_output.token_usage.completion_tokens
            self.total_tokens = crew_output.token_usage.total_tokens
            self.cost = self.cost_calculator.calculate_cost(self.input_tokens, self.output_tokens)
            self.carbon_emissions = calculate_carbon_emissions(self.total_tokens, self.model)
            self.logger.log(f"Input Tokens: {self.input_tokens}")
            self.logger.log(f"Output Tokens: {self.output_tokens}")
            self.logger.log(f"Total Tokens: {self.total_tokens}")
            self.logger.log(f"Cost: ${self.cost:.6f}")
        else:
            self.logger.log("No token usage information available in CrewOutput.")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def end(self):
        self.end_time = time.time()
        if self.enable_monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
        self.total_time = self.end_time - self.start_time
        self.cost = self.cost_calculator.calculate_cost(self.input_tokens, self.output_tokens)
        self.carbon_emissions = calculate_carbon_emissions(self.total_tokens, self.model)
        self.avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        self.throughput = self.total_tokens / self.total_time if self.total_time > 0 else 0

        self.logger.log("AgentMont monitoring ended.")
        self.logger.log(f"Total Time: {self.total_time:.2f} seconds")
        self.logger.log(f"Input Tokens: {self.input_tokens}")
        self.logger.log(f"Output Tokens: {self.output_tokens}")
        self.logger.log(f"Total Tokens: {self.total_tokens}")
        self.logger.log(f"Cost: ${self.cost:.6f}")
        self.logger.log(f"Average Latency: {self.avg_latency:.4f} seconds")
        self.logger.log(f"Throughput (Tokens/Sec): {self.throughput:.2f}")
        self.logger.log(f"CPU Usage: {self.cpu_usage}")
        self.logger.log(f"Memory Usage: {self.memory_usage} MB")
        self.logger.log(f"Carbon Emissions: {self.carbon_emissions:.6f} kg CO2")


class AgentMontExtended(AgentMont):
    def __init__(self, model: str, enable_monitoring: bool = True, encoding_name: str = "cl100k_base"):
        super().__init__(model, enable_monitoring, encoding_name)
        self.visualizer = Visualizer(self)

    def visualize(self, method='cli'):
        if method == 'cli':
            self.visualizer.cli_summary()
        elif method == 'streamlit':
            self.visualizer.streamlit_dashboard()
        else:
            raise ValueError("Unsupported visualization method. Choose 'cli' or 'streamlit'.")

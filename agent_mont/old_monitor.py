# agent_watch/monitor.py

import time
import os
import psutil
import threading
from .cost_calculator import CostCalculator
from .logger import Logger
from .utils import calculate_carbon_emissions
import tiktoken
from .visualizer import Visualizer

class AgentWatch:
    def __init__(self, model: str, enable_monitoring: bool = True):
        self.model = model
        self.start_time = None
        self.end_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost = 0.0
        self.cpu_usage = []
        self.memory_usage = []
        self.logger = Logger()
        self.cost_calculator = CostCalculator(model)
        self.monitoring = False
        self.monitor_thread = None
        self.carbon_emissions = 0.0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Adjust encoding as needed
        self.enable_monitoring = enable_monitoring

    def start(self):
        self.logger.log("Monitoring started.")
        self.start_time = time.time()
        if self.enable_monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.start()

    def end(self):
        self.end_time = time.time()
        if self.enable_monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
        self.total_time = self.end_time - self.start_time
        self.cost = self.cost_calculator.calculate_cost(self.input_tokens, self.output_tokens)
        self.carbon_emissions = calculate_carbon_emissions(self.total_tokens, self.model)
        self.logger.log("Monitoring ended.")
        self.logger.log(f"Total Time: {self.total_time:.2f} seconds")
        self.logger.log(f"Input Tokens: {self.input_tokens}")
        self.logger.log(f"Output Tokens: {self.output_tokens}")
        self.logger.log(f"Total Tokens: {self.total_tokens}")
        self.logger.log(f"Cost: ${self.cost:.6f}")
        self.logger.log(f"CPU Usage: {self.cpu_usage}")
        self.logger.log(f"Memory Usage: {self.memory_usage} MB")
        self.logger.log(f"Carbon Emissions: {self.carbon_emissions:.6f} kg CO2")

    def _monitor_resources(self):
        process = psutil.Process(os.getpid())
        while self.monitoring:
            cpu = process.cpu_percent(interval=1)
            mem = process.memory_info().rss / (1024 * 1024)  # in MB
            self.cpu_usage.append(cpu)
            self.memory_usage.append(mem)
            time.sleep(1)

    def set_token_counts(self, input_text: str, output_text: str):
        self.input_tokens = self.count_tokens(input_text)
        self.output_tokens = self.count_tokens(output_text)
        self.total_tokens = self.input_tokens + self.output_tokens
        self.logger.log(f"Input Tokens: {self.input_tokens}")
        self.logger.log(f"Output Tokens: {self.output_tokens}")
        self.logger.log(f"Total Tokens: {self.total_tokens}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class AgentWatchExtended(AgentWatch):
    def __init__(self, model: str):
        super().__init__(model)
        self.visualizer = Visualizer(self)

    def visualize(self, method='cli'):
        if method == 'cli':
            self.visualizer.cli_summary()
        elif method == 'streamlit':
            self.visualizer.streamlit_dashboard()
        else:
            raise ValueError("Unsupported visualization method. Choose 'cli' or 'streamlit'.")

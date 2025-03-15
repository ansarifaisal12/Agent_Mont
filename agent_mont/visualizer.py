import matplotlib.pyplot as plt
import streamlit as st

class Visualizer:
    def __init__(self, monitor):
        self.monitor = monitor

    def cli_summary(self):
        print("\n--- AgentMont Summary ---")
        print(f"Total Time: {self.monitor.total_time:.2f} seconds")
        print(f"Input Tokens: {self.monitor.input_tokens}")
        print(f"Output Tokens: {self.monitor.output_tokens}")
        print(f"Total Tokens: {self.monitor.total_tokens}")
        print(f"Cost: ${self.monitor.cost:.6f}")
        print(f"Average Latency: {self.monitor.avg_latency:.4f} seconds")
        print(f"Throughput (Tokens/Sec): {self.monitor.throughput:.2f}")
        avg_cpu = sum(self.monitor.cpu_usage) / len(self.monitor.cpu_usage) if self.monitor.cpu_usage else 0
        avg_mem = sum(self.monitor.memory_usage) / len(self.monitor.memory_usage) if self.monitor.memory_usage else 0
        print(f"Average CPU Usage: {avg_cpu:.2f}%")
        print(f"Average Memory Usage: {avg_mem:.2f} MB")
        print(f"Carbon Emissions: {self.monitor.carbon_emissions:.6f} kg CO2")
        print("----------------------------\n")

    def streamlit_dashboard(self):
        st.title("AgentMont Dashboard")
        st.header("Performance Metrics")
        st.write(f"**Total Time:** {self.monitor.total_time:.2f} seconds")
        st.write(f"**Input Tokens:** {self.monitor.input_tokens}")
        st.write(f"**Output Tokens:** {self.monitor.output_tokens}")
        st.write(f"**Total Tokens:** {self.monitor.total_tokens}")
        st.write(f"**Cost:** ${self.monitor.cost:.6f}")
        st.write(f"**Average Latency:** {self.monitor.avg_latency:.4f} seconds")
        st.write(f"**Throughput (Tokens/Sec):** {self.monitor.throughput:.2f}")
        st.write(f"**Carbon Emissions:** {self.monitor.carbon_emissions:.6f} kg CO2")

        st.header("Resource Utilization")
        if self.monitor.cpu_usage and self.monitor.memory_usage:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].plot(self.monitor.cpu_usage, label='CPU Usage (%)', color='blue')
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('CPU Usage (%)')
            ax[0].legend()

            ax[1].plot(self.monitor.memory_usage, label='Memory Usage (MB)', color='orange')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Memory Usage (MB)')
            ax[1].legend()

            st.pyplot(fig)
        else:
            st.write("No resource usage data available.")

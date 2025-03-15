# Agent Mont

<div align="center">
  <img src="https://github.com/ansarifaisal12/Agent_Mont/raw/main/assets/logo.jpeg" alt="Agent Mont Logo" width="150"/>
  <h3>Advanced Monitoring for AI Agents</h3>
  <p>Comprehensive metrics, insights, and visualization for Phidata and Crew AI applications</p>
</div>

  ![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
  ![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)
</div>

---

## 📊 Overview

**Agent Mont** is an advanced AgentOps monitoring library designed specifically for Phidata and Crew AI applications. It provides real-time insights into your AI agents' performance, resource utilization, and environmental impact.

## ✨ Key Features

- **📝 Token Tracking** - Precise accounting of input, output, and total tokens
- **💰 Cost Analysis** - Real-time cost calculation based on model-specific pricing
- **⚡ Performance Metrics** - Detailed execution time and resource utilization
- **🔄 Throughput Monitoring** - Tokens processed per second and operation latency
- **🌱 Environmental Impact** - Carbon emission estimates for responsible AI deployment
- **📋 Comprehensive Logging** - Detailed operational logs for debugging and analysis
- **📊 Interactive Visualization** - CLI summaries and Streamlit dashboard for visual insights

## 🚀 Installation

```bash
pip install agent-mont
```

## 🔍 Usage

### Basic Example

```python
from agent_mont import AgentMontExtended

# Initialize Agent Mont monitoring
mont = AgentMontExtended(model="gpt-4o")

# Start monitoring
mont.start()

# Your Crew AI operations go here
# result = your_ai_function()

# Set token counts
input_text = "Your input text here."
output_text = "Your output text here."
mont.set_token_counts(input_text=input_text, output_text=output_text)

# Measure operation latency
# result = mont.measure_operation(your_function, arg1, arg2)

# End monitoring
mont.end()

# Visualize the results
mont.visualize(method='cli')
```

### Integration with Crew AI

```python
from agent_mont import AgentMontExtended
from crewai import Crew, Agent, Task

# Initialize monitoring
mont = AgentMontExtended(model="gpt-4o")
mont.start()

# Define your Crew agents and tasks
agent = Agent(...)
task = Task(...)
crew = Crew(agents=[agent], tasks=[task])

# Run crew process
result = crew.kickoff()

# Extract token usage from crew output
mont.set_token_usage_from_crew_output(result)

# End monitoring and display metrics
mont.end()
mont.visualize(method='cli')
```

## 📖 API Reference

### AgentMont Initialization

```python
AgentMont(
    model: str,                      # LLM model identifier (e.g., "gpt-4o")
    enable_monitoring: bool = True,  # Toggle monitoring on/off
    encoding_name: str = "cl100k_base"  # Token encoding scheme
)
```

### Core Methods

| Method | Description |
|--------|-------------|
| `start()` | Begin resource and metric monitoring |
| `end()` | Stop monitoring and compute all metrics |
| `set_token_counts(input_text, output_text)` | Set token usage based on text |
| `measure_operation(func, *args, **kwargs)` | Wrap a function call to record its latency |
| `set_token_usage_from_crew_output(crew_output)` | Extract token usage from a CrewOutput |
| `visualize(method)` | Visualize metrics using CLI or Streamlit |

## 📈 Visualization

### CLI Summary
```python
mont.visualize(method='cli')
```

Example output:
```
========== AGENT MONT METRICS ==========
Model: gpt-4o
Execution Time: 2.34s
Total Tokens: 1,245
Cost: $0.0237
Throughput: 532.1 tokens/sec
CPU Usage: 23.7%
Memory Usage: 145.2 MB
Carbon Emissions: 0.00021 gCO2eq
=======================================
```

### Streamlit Dashboard

Create a visualization script:
```python
# visualize.py
from agent_mont import AgentMontExtended

mont = AgentMontExtended(model="gpt-4o")
# Load metrics from logs or set directly
mont.set_token_counts("Example input", "Example output")
mont.end()
mont.visualize(method='streamlit')
```

Run the dashboard:
```bash
streamlit run visualize.py
```

## 📝 Logging

All events and metrics are logged to `agent_mont.log` with timestamps and event categories:

```
2025-03-15 14:32:17 [INFO] Agent Mont initialized with model: gpt-4o
2025-03-15 14:32:18 [INFO] Monitoring started
2025-03-15 14:32:20 [INFO] Token counts set: 145 input, 267 output
2025-03-15 14:32:21 [INFO] Monitoring ended
2025-03-15 14:32:21 [INFO] Metrics calculated: 412 tokens, $0.0082 cost, 1.73s execution time
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Faisal Azeez - [faisalazeeii786@gmail.com](mailto:faisalazeeii786@gmail.com)

---

<div align="center">
  <p>Made with ❤️ for the AI agent community</p>
</div>
import unittest
from agent_mont import AgentMont

class TestAgentMont(unittest.TestCase):
    def test_initialization(self):
        am = AgentMont(model="gpt-4o", enable_monitoring=False)
        self.assertEqual(am.model, "gpt-4o")
        self.assertEqual(am.input_tokens, 0)
        self.assertEqual(am.output_tokens, 0)
        self.assertEqual(am.total_tokens, 0)
        self.assertEqual(am.cost, 0.0)

    def test_cost_calculation_and_metrics(self):
        am = AgentMont(model="gpt-4o", enable_monitoring=False)
        input_text = "Hello world " * 1000
        output_text = "This is a test response " * 500

        am.start()
        am.set_token_counts(input_text, output_text)
        # Simulate recording two operation latencies.
        am.record_latency(0.5)
        am.record_latency(0.7)
        am.end()

        expected_input_tokens = am.count_tokens(input_text)
        expected_output_tokens = am.count_tokens(output_text)
        self.assertEqual(am.input_tokens, expected_input_tokens)
        self.assertEqual(am.output_tokens, expected_output_tokens)
        self.assertAlmostEqual(am.cost, am.cost_calculator.calculate_cost(expected_input_tokens, expected_output_tokens))
        self.assertTrue(am.avg_latency > 0)
        self.assertTrue(am.throughput > 0)

if __name__ == "__main__":
    unittest.main()

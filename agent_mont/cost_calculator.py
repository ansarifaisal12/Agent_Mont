class CostCalculator:
    # Pricing and speed details for various models.
    MODELS_PRICING = {
        "deepseek-r1-distill-llama-70b": {"input_cost": 0.00000075, "output_cost": 0.00000099, "speed": 275},
        "deepseek-r1-distill-qwen-32b": {"input_cost": 0.00000069, "output_cost": 0.00000069, "speed": 128000},
        "qwen-2.5-32b": {"input_cost": 0.00000079, "output_cost": 0.00000079, "speed": 200},
        "qwen-2.5-coder-32b": {"input_cost": 0.00000079, "output_cost": 0.00000079, "speed": 390},
        "qwen-qwq-32b": {"input_cost": 0.00000029, "output_cost": 0.00000039, "speed": 400},
        "mistral-saba-24b": {"input_cost": 0.00000079, "output_cost": 0.00000079, "speed": 330},
        "llama-3.2-1b-preview": {"input_cost": 0.00000004, "output_cost": 0.00000004, "speed": 3100},
        "llama-3.2-3b-preview": {"input_cost": 0.00000006, "output_cost": 0.00000006, "speed": 1600},
        "llama-3.3-70b-versatile": {"input_cost": 0.00000059, "output_cost": 0.00000079, "speed": 275},
        "llama-3.1-8b-instant": {"input_cost": 0.00000005, "output_cost": 0.00000008, "speed": 750},
        "llama3-70b-8192": {"input_cost": 0.00000059, "output_cost": 0.00000079, "speed": 330},
        "llama3-8b-8192": {"input_cost": 0.00000005, "output_cost": 0.00000008, "speed": 1250},
        "mixtral-8x7b-32768": {"input_cost": 0.00000024, "output_cost": 0.00000024, "speed": 575},
        "gemma2-9b-it": {"input_cost": 0.00000020, "output_cost": 0.00000020, "speed": 500},
        "llama-guard-3-8b": {"input_cost": 0.00000020, "output_cost": 0.00000020, "speed": 765},
        "llama-3.3-70b-specdec": {"input_cost": 0.00000059, "output_cost": 0.00000099, "speed": 1600},
        "gpt-4o": {"input_cost": 0.00000250, "output_cost": 0.00001000, "speed": None},
        "gpt-4o-mini": {"input_cost": 0.00000015, "output_cost": 0.00000060, "speed": None},
        "gpt-4": {"input_cost": 0.00000300, "output_cost": 0.00000600, "speed": None},
        "gpt-4o-audio-preview": {"input_cost": 0.00000250, "output_cost": 0.00001000, "speed": None},
        "o1-mini": {"input_cost": 0.00000300, "output_cost": 0.00001200, "speed": None},
        "o1-preview": {"input_cost": 0.00001500, "output_cost": 0.00006000, "speed": None},
        "chatgpt-4o-latest": {"input_cost": 0.00000500, "output_cost": 0.00001500, "speed": None},
        "gpt-4-turbo-preview": {"input_cost": 0.00001000, "output_cost": 0.00003000, "speed": None},
        "gpt-4-32k": {"input_cost": 0.00006000, "output_cost": 0.00012000, "speed": None},
        "gpt-4-turbo": {"input_cost": 0.00001000, "output_cost": 0.00003000, "speed": None},
        "gpt-4-vision-preview": {"input_cost": 0.00001000, "output_cost": 0.00003000, "speed": None},
        "gpt-3.5-turbo": {"input_cost": 0.00000150, "output_cost": 0.00000200, "speed": None},
        "gpt-3.5-turbo-16k": {"input_cost": 0.00000300, "output_cost": 0.00000400, "speed": None},
        "claude-instant-1.2": {"input_cost": 0.00000016, "output_cost": 0.00000055, "speed": None},
        "claude-2": {"input_cost": 0.00000800, "output_cost": 0.00002400, "speed": None},
        "claude-2.1": {"input_cost": 0.00000800, "output_cost": 0.00002400, "speed": None},
        "claude-3-haiku-20240307": {"input_cost": 0.00000025, "output_cost": 0.00000125, "speed": None},
        "claude-3-5-haiku-20241022": {"input_cost": 0.00000100, "output_cost": 0.00000500, "speed": None},
        "claude-3-opus-20240229": {"input_cost": 0.00001500, "output_cost": 0.00007500, "speed": None},
        "claude-3-sonnet-20240229": {"input_cost": 0.00000300, "output_cost": 0.00001500, "speed": None},
        "claude-3-5-sonnet-20240620": {"input_cost": 0.00000300, "output_cost": 0.00001500, "speed": None},
        "claude-3-5-sonnet-20241022": {"input_cost": 0.00000300, "output_cost": 0.00001500, "speed": None},
        "Claude-3.7-Sonnet": {"input_cost": 0.00000300, "output_cost": 0.00001500, "speed": None},
        "Gemini-2.0-Flash": {"input_cost": 0.00000010, "output_cost": 0.00000040, "speed": None},
        "Gemini-2.0=Flash-Lite": {"input_cost": 0.000000075, "output_cost": 0.00000030, "speed": None},
        "gemini-pro": {"input_cost": 0.00000050, "output_cost": 0.00000050, "speed": None},
        "gemini-1.0-pro": {"input_cost": 0.00000050, "output_cost": 0.00000050, "speed": None},
        "gemini-1.5-pro": {"input_cost": 0.00000125, "output_cost": 0.00000500, "speed": None},
        "gemini-1.5-flash": {"input_cost": 0.000000075, "output_cost": 0.00000030, "speed": None},
    }

    def __init__(self, model: str):
        if model not in self.MODELS_PRICING:
            raise ValueError(f"Model {model} not supported.")
        self.model = model
        self.pricing = self.MODELS_PRICING[model]

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_rate = self.pricing["input_cost"]
        output_rate = self.pricing["output_cost"]
        input_cost = input_tokens * (input_rate)
        output_cost = output_tokens * (output_rate)
        return input_cost + output_cost

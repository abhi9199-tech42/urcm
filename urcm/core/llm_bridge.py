import os
import sys
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

class LLMBridge:
    """
    The Neural Interface Layer.
    Connects the AXIOM Cognitive Core (URCM) to a standard LLM (e.g., Phi-2, Mistral).
    """
    
    def __init__(self, model_path: str = None, n_ctx: int = 2048):
        self.model = None
        self.mock_mode = False
        self.fail_count = 0
        self.breaker_open = False
        self.breaker_threshold = int(os.getenv("LLM_BREAKER_THRESHOLD", "3"))
        self.breaker_cooldown_sec = int(os.getenv("LLM_BREAKER_COOLDOWN", "10"))
        self._breaker_last_open = 0
        
        if not HAS_LLAMA:
            print("⚠️ [LLMBridge] 'llama-cpp-python' not found. Running in MOCK mode.")
            self.mock_mode = True
            return

        if model_path and os.path.exists(model_path):
            expected_sha = os.getenv("LLM_MODEL_SHA256")
            if expected_sha:
                import hashlib
                sha = hashlib.sha256()
                with open(model_path, "rb") as mf:
                    for chunk in iter(lambda: mf.read(8192), b""):
                        sha.update(chunk)
                digest = sha.hexdigest()
                if digest.lower() != expected_sha.lower():
                    print(f"[LLMBridge] ❌ Integrity check failed: expected {expected_sha}, got {digest}")
                    self.mock_mode = True
                    return
            print(f"[LLMBridge] 🔌 Loading Model: {model_path} ...")
            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    verbose=False
                )
                print("[LLMBridge] ✅ Model Loaded.")
            except Exception as e:
                print(f"[LLMBridge] ❌ Failed to load model: {e}")
                self.mock_mode = True
        else:
            print(f"[LLMBridge] ⚠️ Model path '{model_path}' not found. Running in MOCK mode.")
            self.mock_mode = True

    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 150, 
                 temperature: float = 0.7,
                 stop: list = None) -> str:
        """
        Generates text completion.
        """
        if self.mock_mode:
            return f" [MOCK LLM RESPONSE] I see you are asking about: {prompt[:50]}... (Model not loaded)"
        import time
        if self.breaker_open:
            if time.time() - self._breaker_last_open < self.breaker_cooldown_sec:
                return " [LLMBridge] Circuit open; skipping generation to prevent cascading failures."
            else:
                self.breaker_open = False
        attempts = 0
        last_error = None
        while attempts < 3:
            try:
                output = self.model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or ["User:", "\n\n"]
                )
                # reset fail count on success
                self.fail_count = 0
                return output['choices'][0]['text']
            except Exception as e:
                last_error = e
                self.fail_count += 1
                attempts += 1
                time.sleep(0.5 * attempts)
        if self.fail_count >= self.breaker_threshold:
            self.breaker_open = True
            self._breaker_last_open = time.time()
        return f" [LLMBridge] Error generating completion: {last_error}"

    def stream(self, 
               prompt: str, 
               max_tokens: int = 150, 
               temperature: float = 0.7,
               stop: list = None):
        """
        Streams text completion.
        """
        if self.mock_mode:
            yield " [MOCK] Streaming response..."
            return
        import time
        if self.breaker_open and (time.time() - self._breaker_last_open < self.breaker_cooldown_sec):
            yield " [LLMBridge] Circuit open; streaming disabled temporarily."
            return
        attempts = 0
        while attempts < 3:
            try:
                stream = self.model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or ["User:", "\n\n"],
                    stream=True
                )
                self.fail_count = 0
                for output in stream:
                    yield output['choices'][0]['text']
                return
            except Exception as e:
                self.fail_count += 1
                attempts += 1
                time.sleep(0.5 * attempts)
        if self.fail_count >= self.breaker_threshold:
            self.breaker_open = True
            self._breaker_last_open = time.time()
        yield " [LLMBridge] Error streaming completion; circuit open."

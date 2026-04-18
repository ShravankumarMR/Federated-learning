from typing import Any


class RAGAgent:
    def retrieve(self, query: str) -> dict[str, Any]:
        # Placeholder retrieval layer. Replace with vector DB integration.
        if not query.strip():
            return {"context": [], "confidence": 0.0}

        return {
            "context": [
                "Historical fraud pattern: bursty session timings",
                "Biometric drift can indicate account sharing",
            ],
            "confidence": 0.72,
        }

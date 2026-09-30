import requests
import json
import logging
import os
import numpy as np
from typing import Dict, Any, List, Optional
from config import KSERVE_ENDPOINT_URL, MODEL_NAME, SYSTEM_ONE_DECISION_OPTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemOneScorerClient:
    """
    Client for pngwn/system-one-qwen3.5-4b-scorer on vLLM.
    """
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        options: Optional[List[str]] = None
    ):
        self._custom_endpoint_url = endpoint_url
        self._custom_model_name = model_name
        self.options = options or SYSTEM_ONE_DECISION_OPTIONS

    @property
    def endpoint_url(self) -> str:
        if self._custom_endpoint_url:
            return self._custom_endpoint_url
        return os.getenv("KSERVE_ENDPOINT_URL", os.getenv("APP_KSERVE_ENDPOINT_URL", KSERVE_ENDPOINT_URL))

    @property
    def model_name(self) -> str:
        if self._custom_model_name:
            return self._custom_model_name
        return os.getenv("MODEL_NAME", os.getenv("APP_MODEL_NAME", MODEL_NAME))

    def _discover_model_name(self, base_url: str) -> str:
        """Query /v1/models on the server to retrieve the exact model ID registered in vLLM."""
        try:
            models_url = base_url.rsplit("/v1/", 1)[0] + "/v1/models"
            resp = requests.get(models_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    first_model = data["data"][0].get("id")
                    if first_model:
                        logger.info(f"Discovered active model ID from KServe: '{first_model}'")
                        return first_model
        except Exception:
            pass
        return self.model_name

    def score_transaction(self, context_card: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a decision scoring request to vLLM.
        Returns probabilities across target options and the winning risk classification.
        """
        if prompt is not None:
            final_prompt = prompt
        elif "System One" in context_card or "Options:" in context_card or "Answer:" in context_card:
            final_prompt = context_card
        else:
            final_prompt = (
                f"Context:\n{context_card}\n\n"
                f"Question: What is the fraud risk classification for this transaction?\n"
                f"Options: {', '.join(self.options)}\n"
                f"Answer: "
            )

        candidates = [self.endpoint_url]
        errors = []
        for url in candidates:
            target_model = self._discover_model_name(url)
            payload = {
                "model": target_model,
                "prompt": final_prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 5
            }

            logger.info(f"Posting decision request to model at {url} (model='{target_model}')...")
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                if response.status_code == 200:
                    self._custom_endpoint_url = url
                    data = response.json()
                    res = self._parse_kserve_response(data)
                    res["backend_url"] = url
                    res["model"] = target_model
                    return res
                else:
                    err_msg = f"HTTP {response.status_code} from {url}: {response.text[:200]}"
                    errors.append(err_msg)
                    logger.warning(err_msg)
            except requests.exceptions.RequestException as req_err:
                err_msg = f"Connection error for {url}: {req_err}"
                errors.append(err_msg)
                logger.warning(err_msg)

        all_err_details = " | ".join(errors)
        raise RuntimeError(f"Failed to reach inference backend after trying candidates: {all_err_details}")

    def _parse_kserve_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses logprob completion responses from vLLM /v1/completions API.
        """
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "logprobs" in choice and choice["logprobs"] and "top_logprobs" in choice["logprobs"]:
                top_lp = choice["logprobs"]["top_logprobs"][0]
                raw_exp = {}
                for idx, opt in enumerate(self.options):
                    idx_str = str(idx)
                    val = None
                    for token_str, logprob_val in top_lp.items():
                        clean_tok = token_str.strip()
                        if clean_tok == idx_str or clean_tok == opt or clean_tok in opt:
                            val = logprob_val
                            break
                    if val is None:
                        val = -10.0
                    raw_exp[opt] = np.exp(val)
                total = sum(raw_exp.values()) or 1.0
                probs = {k: round(float(v / total), 4) for k, v in raw_exp.items()}
            else:
                text_out = choice.get("text", "").strip()
                probs = {opt: (0.8 if opt in text_out else 0.1) for opt in self.options}
        best_category = response_json.get("prediction", max(probs, key=probs.get))
        fraud_score = float(probs.get("HIGH_RISK_FRAUD", 0.0) + 0.5 * probs.get("MEDIUM_RISK", 0.0))

        return {
            "prediction": best_category,
            "probabilities": probs,
            "fraud_risk_score": round(fraud_score, 4),
            "backend": "vLLM"
        }

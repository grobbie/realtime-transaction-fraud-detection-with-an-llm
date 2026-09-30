from typing import Dict, Any, List, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import PromptTemplate
from data_loader import FraudDataLoader
from system_one_client import SystemOneScorerClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAUD_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """<|im_start|>user
System One Fraud Risk Decision Query:

Target Transaction Profile:
{context_card}

Question: What is the fraud risk classification for this transaction?
Options: 0: LOW_RISK, 1: MEDIUM_RISK, 2: HIGH_RISK_FRAUD<|im_end|>
<|im_start|>assistant
Answer: """
)

class SystemOneFraudChain(Runnable):
    """
    LangChain Runnable implementation for Transaction Fraud Risk Scoring using
    pngwn/system-one-qwen3.5-4b-scorer on KServe.
    """
    def __init__(self, client: Optional[SystemOneScorerClient] = None):
        self.client = client or SystemOneScorerClient()
        self.prompt = FRAUD_PROMPT_TEMPLATE

    def invoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Processes a transaction dictionary through LangChain context formatting
        and System One non-autoregressive decision scoring.
        """
        # Step 1: Format context card if not already formatted
        if "context_card" in input_data:
            context_card = input_data["context_card"]
        else:
            context_card = FraudDataLoader.format_context_card(input_data)

        # Step 2: Format prompt via LangChain template
        formatted_prompt = self.prompt.format(context_card=context_card)

        # Step 3: Run System One decision scoring
        scoring_res = self.client.score_transaction(context_card=context_card, prompt=formatted_prompt)

        # Step 4: Construct decision rationale
        risk_category = scoring_res["prediction"]
        risk_score = scoring_res["fraud_risk_score"]
        probs = scoring_res["probabilities"]

        if risk_category == "HIGH_RISK_FRAUD":
            rationale = "CRITICAL RISK: High probability of account drain anomaly and recipient zero-balance anomaly detected by System One scorer."
        elif risk_category == "MEDIUM_RISK":
            rationale = "MODERATE RISK: Elevated transaction amount relative to recipient history. Secondary review recommended."
        else:
            rationale = "LOW RISK: Standard financial transaction pattern with normal balance deltas."

        return {
            "input_transaction": input_data,
            "context_card": context_card,
            "prediction": risk_category,
            "fraud_risk_score": risk_score,
            "probabilities": probs,
            "decision_rationale": rationale,
            "backend": scoring_res.get("backend", "System One KServe")
        }


import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDataLoader:
    def __init__(self):
        pass

    def load_dataset(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Generates realistic PaySim financial transactions locally for batch evaluation.
        """
        logger.info(f"Generating {sample_size} PaySim financial transaction records locally.")
        return self._generate_synthetic_paysim(sample_size=sample_size)

    def _generate_synthetic_paysim(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Generates realistic PaySim financial transactions with typical fraud anomalies
        (e.g., account drains via TRANSFER or CASH_OUT, large amounts, empty recipient accounts).
        """
        np.random.seed(42)
        n_fraud = max(1, int(sample_size * 0.15))
        n_legit = sample_size - n_fraud

        types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        
        # Legit transactions
        legit_data = []
        for _ in range(n_legit):
            t_type = np.random.choice(types, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            amount = round(float(np.random.exponential(scale=2500.0) + 10.0), 2)
            old_org = round(float(np.random.uniform(500, 50000)), 2)
            new_org = max(0.0, round(old_org - amount, 2)) if t_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"] else round(old_org + amount, 2)
            old_dest = round(float(np.random.uniform(100, 30000)), 2)
            new_dest = round(old_dest + amount, 2)
            legit_data.append({
                "step": np.random.randint(1, 100),
                "type": t_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": 0
            })

        # Fraudulent transactions (classic PaySim fraud patterns)
        fraud_data = []
        for _ in range(n_fraud):
            t_type = np.random.choice(["TRANSFER", "CASH_OUT"])
            old_org = round(float(np.random.uniform(50000, 500000)), 2)
            amount = old_org  # Account drain pattern!
            new_org = 0.0     # Account wiped out
            old_dest = 0.0    # New/unknown account
            new_dest = 0.0 if t_type == "CASH_OUT" else amount
            fraud_data.append({
                "step": np.random.randint(1, 100),
                "type": t_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "isFraud": 1
            })

        all_records = legit_data + fraud_data
        df = pd.DataFrame(all_records).sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df

    @staticmethod
    def format_context_card(row: Dict[str, Any]) -> str:
        """
        Converts a transaction row into a formatted, human-readable & LLM-parseable transaction context card.
        """
        t_type = row.get("type", "TRANSFER")
        amount = row.get("amount", 0.0)
        old_org = row.get("oldbalanceOrg", 0.0)
        new_org = row.get("newbalanceOrig", 0.0)
        old_dest = row.get("oldbalanceDest", 0.0)
        new_dest = row.get("newbalanceDest", 0.0)

        org_delta = old_org - new_org
        dest_delta = new_dest - old_dest

        context = (
            f"TRANSACTION PROFILE:\n"
            f"- Type: {t_type}\n"
            f"- Transaction Amount: ${amount:,.2f}\n"
            f"- Sender Initial Balance: ${old_org:,.2f}\n"
            f"- Sender Post-Transaction Balance: ${new_org:,.2f} (Delta: ${org_delta:,.2f})\n"
            f"- Recipient Initial Balance: ${old_dest:,.2f}\n"
            f"- Recipient Post-Transaction Balance: ${new_dest:,.2f} (Delta: ${dest_delta:,.2f})"
        )
        return context

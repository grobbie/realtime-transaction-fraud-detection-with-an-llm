import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import sys
import os

# Ensure the module directory (/app) is at the front of sys.path for local imports
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from fraud_chain import SystemOneFraudChain
from config import MODEL_NAME, KSERVE_ENDPOINT_URL


app = FastAPI(
    title="Transaction Fraud Risk Scoring API",
    description="Transaction Fraud Risk Scoring application",
    version="1.0.0"
)

chain = SystemOneFraudChain()

class TransactionRequest(BaseModel):
    step: int = Field(default=1, description="Simulation step / hour")
    type: str = Field(default="TRANSFER", description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, etc.")
    amount: float = Field(default=150000.0, description="Transaction dollar amount")
    oldbalanceOrg: float = Field(default=150000.0, description="Sender balance before transaction")
    newbalanceOrig: float = Field(default=0.0, description="Sender balance after transaction")
    oldbalanceDest: float = Field(default=0.0, description="Recipient balance before transaction")
    newbalanceDest: float = Field(default=0.0, description="Recipient balance after transaction")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "kserve_endpoint": KSERVE_ENDPOINT_URL,
    }

@app.post("/score")
def score_transaction(txn: TransactionRequest):
    try:
        txn_dict = txn.model_dump()
        result = chain.invoke(txn_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index_ui():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transaction Fraud Risk Scoring Engine</title>
        <!-- Vanilla Framework CSS -->
        <link rel="stylesheet" href="https://assets.ubuntu.com/v1/vanilla-framework-version-4.18.0.min.css" />
        <!-- Ubuntu Sans & Ubuntu Sans Mono Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Ubuntu+Sans:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Ubuntu+Sans+Mono:wght@400;600&display=swap" rel="stylesheet">
        
        <style>
            :root {{
                --theme-aubergine: #77216F;
                --theme-dark: #2C001E;
                --ubuntu-orange: #E95420;
                --font-sans: 'Ubuntu Sans', system-ui, -apple-system, sans-serif;
                --font-mono: 'Ubuntu Sans Mono', monospace;
            }}
            body {{
                font-family: var(--font-sans) !important;
                background-color: #F7F7F7;
                color: #111111;
                margin: 0;
            }}
            .p-navigation {{
                background-color: #111111;
                border-bottom: 3px solid var(--ubuntu-orange);
            }}
            .brand-title {{
                color: #ffffff;
                font-weight: 700;
                font-size: 1.25rem;
                letter-spacing: -0.02em;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}
            .brand-circle {{
                width: 14px;
                height: 14px;
                background-color: var(--ubuntu-orange);
                border-radius: 50%;
                display: inline-block;
            }}
            .p-panel {{
                background: #ffffff;
                border: 1px solid #E5E5E5;
                border-radius: 4px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }}
            .p-chip {{
                font-family: var(--font-mono);
                font-size: 0.85rem;
            }}
            .result-terminal {{
                background-color: #111111;
                color: #00FF66;
                font-family: var(--font-mono);
                padding: 1rem;
                border-radius: 4px;
                border-left: 4px solid var(--ubuntu-orange);
                white-space: pre-wrap;
                font-size: 0.85rem;
                max-height: 400px;
                overflow-y: auto;
            }}
            .p-button--brand {{
                background-color: var(--ubuntu-orange) !important;
                border-color: var(--ubuntu-orange) !important;
                color: #ffffff !important;
                font-family: var(--font-sans);
                font-weight: 600;
            }}
            .p-button--brand:hover {{
                background-color: #D34311 !important;
            }}
            .risk-pill-badge {{
                display: inline-block;
                padding: 0.4rem 1.2rem;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                font-family: var(--font-sans);
                margin-bottom: 1rem;
            }}
            .HIGH_RISK_FRAUD {{ background-color: #C7162B; color: #FFFFFF; }}
            .MEDIUM_RISK {{ background-color: #F99B11; color: #111111; }}
            .LOW_RISK {{ background-color: #0E8420; color: #FFFFFF; }}
            .code-inline {{
                font-family: var(--font-mono);
                background-color: #E5E5E5;
                padding: 0.1rem 0.4rem;
                border-radius: 3px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <!-- Main Content Layout -->
        <main class="u-fixed-width" style="margin-top: 2rem; margin-bottom: 4rem;">
            <div class="row">
                <!-- Transaction Inputs Column -->
                <div class="col-6">
                    <div class="p-panel">
                        <h3 class="p-heading--4" style="border-bottom: 2px solid #E5E5E5; padding-bottom: 0.5rem;">Simulate Transaction</h3>
                        <form id="scoreForm">
                            <div class="p-form__group">
                                <label class="p-form__label" for="type">Transaction Type</label>
                                <select class="p-form__control" id="type">
                                    <option value="TRANSFER" selected>TRANSFER (High Anomaly Potential)</option>
                                    <option value="CASH_OUT">CASH_OUT</option>
                                    <option value="PAYMENT">PAYMENT</option>
                                    <option value="DEBIT">DEBIT</option>
                                </select>
                            </div>
                            <div class="p-form__group">
                                <label class="p-form__label" for="amount">Transaction Amount ($)</label>
                                <input class="p-form__control" type="number" id="amount" value="250000" step="1000">
                            </div>
                            <div class="p-form__group">
                                <label class="p-form__label" for="oldbalanceOrg">Sender Initial Balance ($)</label>
                                <input class="p-form__control" type="number" id="oldbalanceOrg" value="250000">
                            </div>
                            <div class="p-form__group">
                                <label class="p-form__label" for="newbalanceOrig">Sender Post-Transaction Balance ($)</label>
                                <input class="p-form__control" type="number" id="newbalanceOrig" value="0">
                            </div>
                            <div class="p-form__group">
                                <label class="p-form__label" for="oldbalanceDest">Recipient Initial Balance ($)</label>
                                <input class="p-form__control" type="number" id="oldbalanceDest" value="0">
                            </div>
                            <div class="p-form__group">
                                <label class="p-form__label" for="newbalanceDest">Recipient Post-Transaction Balance ($)</label>
                                <input class="p-form__control" type="number" id="newbalanceDest" value="0">
                            </div>
                            <button type="button" class="p-button--brand u-no-margin--bottom" style="width: 100%; margin-top: 1rem;" onclick="submitScore()">Evaluate Risk Score</button>
                        </form>
                    </div>
                </div>

                <!-- Decision Output Column -->
                <div class="col-6">
                    <div class="p-panel" style="min-height: 100%;">
                        <h3 class="p-heading--4" style="border-bottom: 2px solid #E5E5E5; padding-bottom: 0.5rem;">System One Decision Output</h3>
                        <div id="outputPill"></div>
                        <div class="result-terminal" id="outputJson">Submit a transaction record to evaluate system risk probabilities.</div>
                    </div>
                </div>
            </div>
        </main>

        <script>
            async function submitScore() {{
                const payload = {{
                    type: document.getElementById('type').value,
                    amount: parseFloat(document.getElementById('amount').value),
                    oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value),
                    newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value),
                    oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value),
                    newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value)
                }};

                document.getElementById('outputJson').innerText = "Querying System One decision scorer...";
                
                try {{
                    const resp = await fetch('/score', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }});
                    const rawText = await resp.text();
                    let data;
                    try {{
                        data = JSON.parse(rawText);
                    }} catch (jsonErr) {{
                        throw new Error(`Server returned HTTP ${{resp.status}} (${{resp.statusText}}): ${{rawText}}`);
                    }}
                    if (!resp.ok) {{
                        const detail = data.detail || data.message || JSON.stringify(data);
                        throw new Error(`HTTP ${{resp.status}}: ${{detail}}`);
                    }}
                    
                    const pillClass = data.prediction || 'LOW_RISK';
                    document.getElementById('outputPill').innerHTML = `<div class="risk-pill-badge ${{pillClass}}">${{data.prediction}} (Score: ${{data.fraud_risk_score}})</div>`;
                    document.getElementById('outputJson').innerText = JSON.stringify(data, null, 2);
                }} catch (e) {{
                    document.getElementById('outputPill').innerHTML = `<div class="risk-pill-badge HIGH_RISK_FRAUD">SCORING ERROR</div>`;
                    document.getElementById('outputJson').innerText = "Error: " + e.message;
                }}
            }}

        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("===============================================================")
    print(" Starting Transaction Fraud Risk Scoring Engine")
    print(f" Model: {MODEL_NAME}")
    print("===============================================================")
    
    print("\n[FastAPI Web App] Launching Uvicorn server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

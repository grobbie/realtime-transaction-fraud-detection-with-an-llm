# Realtime transaction fraud detection – with an LLM?

[![Blog Post](https://img.shields.io/badge/Blog_Post-Read_Tutorial-blue?style=for-the-badge&logo=medium)](https://ubuntu.com/blog/realtime-transaction-fraud-detection-with-an-LLM)

> **Accompanying Tutorial**: For a detailed step-by-step walkthrough, design decisions, and deep-dive explanations, read the complete article: **[Link to Blog Post](https://ubuntu.com/blog/realtime-transaction-fraud-detection-with-an-LLM)**.

This repository contains the complete code and Kubernetes manifests for preparing and running a Jev-like LLM behind the firewall, and then using it to assess banking transactions for risk of fraud.

---

1. **Model merge & quantization**: Merge LoRA adapter weights into base model and quantize to shrink the model so that it can run on a laptop.
2. **Run the model**: Deploy the model with KServe on vLLM.
3. **Test the model on the command line**: Run a command line test using `curl`.
4. **Run a webUI**: Run a simple webUI to help visualize fraud risk scoring.

---

## Repository file structure

| File | Description |
| :--- | :--- |
| [`app.py`](app.py) | A webUI for visually testing the fraud detection. |
| [`config.py`](config.py) | Config file for the webUI. |
| [`data_loader.py`](data_loader.py) | Utilities for formatting and generating transaction data. |
| [`fraud_chain.py`](fraud_chain.py) | A LangChain flow that structures interaction with the LLM. |
| [`kserve_deployment_cpu.yaml`](kserve_deployment_cpu.yaml) | Creates the LLM model server deployment on a K8s cluster. |
| [`kserve_service_lb.yaml`](kserve_service_lb.yaml) | Creates a loadbalancer/ingress for the LLM model server so you can talk to it over the network. |
| [`quantize_awq.py`](quantize.awq.py) | Merges the LoRA adapter weights and quantizes the model for more efficient running on constrained devices. |
| [`requirements.txt`](requirements.txt) | The Python dependencies for the project. |
| [`system_one_client.py`](system_one_client.py) | Tools library class for interacting with the model server. |

---

## References

* **Blog Post Tutorial**: [Read the full article](https://ubuntu.com/blog/realtime-transaction-fraud-detection-with-an-LLM)
* [Discover the full capabilities of Charmed Kubeflow](https://canonical.com/mlops/kubeflow)
* [KServe Documentation](https://kserve.github.io/website/)

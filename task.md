# Task Management: Intelligent Omnichannel Chatbot

This file tracks the status of all project components, implementation steps, and missing features.

## Project Roadmap

### Phase 1: Infrastructure & Core RAG (Current)
*   **Goal:** Establish the foundation for a single-store functional bot.
*   **Status:** ~90% Complete.

| Task | Status | Notes |
| :--- | :--- | :--- |
| Self-hosted Chatwoot Deployment | [x] Done | Running on Proxmox. |
| FastAPI Webhook Handler | [x] Done | HMAC verification & event parsing implemented. |
| Basic PII Masking Service | [x] Done | Regex-based (Phone, Email, Name, Address). |
| Basic Qdrant Search Service | [x] Done | Single-collection retrieval. |
| GPT-4o Integration | [x] Done | Grounded reply logic in `llm_service.py`. |
| Basic Escalation Logic | [x] Done | Keyword-based and low-confidence triggers. |
| Logging & Observability | [x] Done | Structured JSON event logging for key steps. |

### Phase 2: Advanced RAG & Fine-tuning
*   **Goal:** Improve intelligence, language quality, and knowledge handling.
*   **Status:** Pending.

| Task | Status | Notes |
| :--- | :--- | :--- |
| LlamaIndex Integration | [x] Done | LlamaIndex-based retrieval added in `rag_service.py`. |
| Hierarchical Chunking | [x] Done | Hierarchical ingestion pipeline added in `ingestion_service.py`. |
| Fine-tuning Dataset Prep | [x] Done | Strict filter, review-first CSV with status, and compiler to training_dataset.jsonl. |
| Qwen 2.5 Fine-tuning | [ ] Missing | Optimize for Tunisian Darija register. |
| Evaluation Framework | [x] Done | Automated & human (agent) blind comparison tests. |

### Phase 3: Multi-store & Business Ops
*   **Goal:** Scale to N stores and automate commerce tasks.
*   **Status:** In Progress.

| Task | Status | Notes |
| :--- | :--- | :--- |
| Store Resolution Service | [x] Done | Maps `inbox_id` to `StoreContext`. |
| WooCommerce API Wrapper | [x] Done | Search, Price/Stock, Draft Order implemented. |
| Developer `/wc` Commands | [x] Done | Allows manual testing of Woo integration via chat. |
| Autonomous Tool Calling | [x] Done | OpenAI function calling integrated into `llm_service.py` to trigger WooCommerce rules natively based on context. |
| Active Learning Loop | [x] Done | Logs low-confidence retrieval events to PostgreSQL via `active_learning_service.py`. |
| Metabase Dashboards | [ ] Missing | Visualize AI metrics from PostgreSQL. |

### Phase 4: Local Qwen 2.5 Fine-tuning Preparation
*   **Goal:** Prepare the full fine-tuning pipeline locally without starting training.
*   **Status:** Complete.

| Task | Status | Notes |
| :--- | :--- | :--- |
| Dataset Validation | [x] Done | JSONL validator with malformed-example detection and distribution reporting. |
| Dataset Split | [x] Done | Reproducible 80/10/10 split into train/validation/test. |
| Review-First Preparation | [x] Done | Review CSV compiled into `training_dataset.jsonl` before validation/split. |
| System Prompt | [x] Done | ROOT4PRO support prompt in `training/prompts/system_prompt.txt`. |
| LoRA / QLoRA Configs | [x] Done | Training presets in `training/configs/`. |
| Training Config | [x] Done | YAML with base model, LR, epochs, batch size, accumulation, and logging. |
| Inference Script | [x] Done | CPU-safe interactive CLI with optional LoRA adapter loading. |
| Benchmark Script | [x] Done | Base vs fine-tuned comparison outputs for later review. |
| Documentation | [x] Done | Complete usage docs in `training/README.md`. |
| Training Execution | [ ] Missing | Deferred to the future GPU server run. |

---

## Component Details

### 1. Chatbot Backend (`chatbot-backend/`)
*   **State:** Functional core.
*   **Missing:** Active learning loop, evaluation harness.
*   **Dependencies:** Qdrant, OpenAI, WooCommerce API.

### 2. Infrastructure (`infra/`)
*   **State:** Docker-ready.
*   **Missing:** Metabase container.
*   **Note:** Runs on Proxmox environment.

### 3. PII Service (`app/services/pii_service.py`)
*   **State:** Regex/Heuristic based.
*   **Improvement:** Consider lightweight NER (spaCy/CAMeL) for better Name/Address detection in Darija.

### 4. WooCommerce Service (`app/services/woocommerce_service.py`)
*   **State:** Clean wrapper over REST API.
*   **Missing:** Webhook handlers for product updates (to trigger re-indexing).

---

## Next Steps (Immediate)

1.  **GPU Handoff:** Move `training/` artifacts to the GPU server and run training later when the compute environment is ready.

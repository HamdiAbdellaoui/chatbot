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
| LlamaIndex Integration | [ ] Missing | Replace basic Qdrant search with LlamaIndex for better chunking. |
| Hierarchical Chunking | [ ] Missing | Implement Section Summary + Full Text levels. |
| Fine-tuning Dataset Prep | [ ] Missing | Export Meta/Mattermost history and clean PII. |
| Qwen 2.5 Fine-tuning | [ ] Missing | Optimize for Tunisian Darija register. |
| Evaluation Framework | [ ] Missing | Automated & human (agent) blind comparison tests. |

### Phase 3: Multi-store & Business Ops
*   **Goal:** Scale to N stores and automate commerce tasks.
*   **Status:** In Progress.

| Task | Status | Notes |
| :--- | :--- | :--- |
| Store Resolution Service | [x] Done | Maps `inbox_id` to `StoreContext`. |
| WooCommerce API Wrapper | [x] Done | Search, Price/Stock, Draft Order implemented. |
| Developer `/wc` Commands | [x] Done | Allows manual testing of Woo integration via chat. |
| Autonomous Tool Calling | [ ] Missing | Let LLM decide when to call WooCommerce (requires OpenAI Tools). |
| Active Learning Loop | [ ] Missing | Flag low-confidence responses for supervisor review. |
| Metabase Dashboards | [ ] Missing | Visualize AI metrics from PostgreSQL. |

---

## Component Details

### 1. Chatbot Backend (`chatbot-backend/`)
*   **State:** Functional core.
*   **Missing:** Integration of LlamaIndex and Tool Calling.
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
1.  **LlamaIndex Upgrade:** Transition `rag_service.py` to use LlamaIndex for more robust retrieval.
2.  **Tool Calling:** Update `llm_service.py` and `chatbot_service.py` to use OpenAI Function Calling for WooCommerce tasks instead of manual `/wc` commands.
3.  **Active Learning:** Implement the PostgreSQL logging for low-confidence flags to feed the human review queue.

# GEMINI.md - Intelligent Omnichannel Chatbot

This document provides architectural mandates, service maps, and development workflows for the AI Agent (Gemini CLI) working on this project.

## 1. Project Identity & Vision
An intelligent omnichannel multi-store chatbot serving Tunisian e-commerce platforms. It centralizes WhatsApp, Instagram, FB Messenger, and Web chat via **Chatwoot**, uses **RAG** (Qdrant) for knowledge, and integrates with **WooCommerce** for real-time product data and draft orders.

**Target File:** `task.md` (Link: [task.md](./task.md)) - Tracks detailed component status and next steps.

---

## 2. Architectural Mandates (Non-Negotiable)

### 2.1 Multi-Store Context Isolation
*   Every conversation MUST be mapped to a `StoreContext` via `store_context_service.py`.
*   The mapping is derived from `inbox_id` or `inbox_name` provided in the Chatwoot webhook.
*   **Mandate:** Never perform a Qdrant search or WooCommerce call without a resolved `store_id`. Use the store-specific collection/credentials.

### 2.2 Privacy & PII Boundary
*   PII (Personally Identifiable Information) MUST be masked before sending data to any external LLM API (OpenAI).
*   **Workflow:** `detect_pii` -> `mask_pii` -> `LLM call`.
*   **Exception:** RAG retrieval (Qdrant) uses the raw query for accuracy, as Qdrant is hosted locally on Proxmox.

### 2.3 WooCommerce Safety
*   Bot-initiated write operations MUST only create **Draft Orders** with status `pending`.
*   **Mandate:** Never allow the bot to confirm payments or delete records. All write actions require human verification in the WooCommerce CRM.

### 2.4 Language Mirroring
*   The bot MUST mirror the client's language (Tunisian Darija, French, or Standard Arabic).
*   LLM instructions (System Prompt) are the primary mechanism for this in Phase 1.

---

## 3. Core Services Map

| Service | Responsibility | State |
| :--- | :--- | :--- |
| `ChatbotService` | Pipeline orchestrator (`chatbot_service.py`). | Core logic implemented. |
| `ChatwootService` | API communication with Chatwoot (messages, labels). | Functional. |
| `PIIService` | Regex-based PII detection and masking. | Functional. |
| `RAGService` | Qdrant retrieval and context formatting. | Basic (LlamaIndex pending). |
| `WooCommerceService` | Client for WooCommerce REST API. | Functional for search/get/draft. |
| `StoreContextService` | Config-driven store resolution (`settings.STORES_JSON`). | Robust implementation. |
| `EscalationService` | Detects human handoff requests or low confidence. | Functional. |

---

## 4. Development & Validation Workflows

### 4.1 Local Development
*   Backend: `FastAPI` running on port 8000.
*   Config: Managed via `.env` and `app/config.py`.
*   **Mandate:** Use `pydantic-settings` for all configuration.

### 4.2 Verification Scripts
Before committing changes to core services, run the relevant script in `chatbot-backend/scripts/`:
*   `test_llm.py`: Verify OpenAI connectivity and PII masking.
*   `test_rag_llm.py`: Verify the RAG + LLM grounded response chain.
*   `qdrant_demo.py`: Verify Vector DB connectivity and indexing.

### 4.3 Adding New Features
1.  Check `task.md` for current progress.
2.  Update `app/config.py` if new environment variables are needed.
3.  Implement service logic with comprehensive logging (following the JSON event log pattern in `chatbot_service.py`).
4.  Expose via `chatbot_service.py` and test with a script.

---

## 5. Security Standards
*   **Secrets:** Never log `OPENAI_API_KEY` or WooCommerce secrets.
*   **PII Debugging:** Use `settings.PII_DEBUG=True` to log spans/confidence, but NEVER raw values.
*   **Database:** PostgreSQL and Qdrant are bound to `localhost` within the Proxmox/Docker environment.

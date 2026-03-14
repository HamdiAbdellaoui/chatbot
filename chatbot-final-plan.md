# Chatbot Intelligent Métier — Final Plan

> **Status:** Complete — ready for implementation planning
> **Source:** `chatbot-initial-plan.md`
> **Language:** English

---

## 1. Project Overview

### 1.1 Vision
An intelligent omnichannel business chatbot serving a **multi-store platform** (currently `ingcoofficiel.tn`, `technotools.tn`, and growing) across WhatsApp, Instagram, Facebook Messenger, and each store's website. One shared bot instance handles all stores, is store-aware per conversation, understands Tunisian Darija/French/Arabic, retrieves knowledge via RAG, manages WooCommerce orders, and hands off seamlessly to live agents via Chatwoot.

### 1.2 Core Objectives
- Automate customer responses across 4 channels from a single system
- Understand customer intent in Standard Arabic, French, and Tunisian Darija
- Answer accurately using RAG over product catalog and internal documents
- Create draft WooCommerce orders on behalf of clients (reviewed by human)
- Route to live agents when requested, fall back to async ticket when offline
- Continuously improve via fine-tuning on real conversation history

### 1.3 Out of Scope (v1)
- Integrated payment processing
- Standalone mobile app
- Advanced multilingual translation beyond Arabic/French/Darija

---

## 2. Target Users & Personas

| Persona | Description | Key Needs |
|---|---|---|
| **End Customer** | Sends queries via any channel | Fast answers, 24/7 availability, choice of bot or human |
| **Sales Agent / Support** | Handles live chats and escalations via Chatwoot | Full conversation context, easy handoff |
| **Administrator** | Manages RAG knowledge base, monitors bot quality | Dashboard, document upload, confidence logs |

---

## 3. Channels & Routing

### 3.1 Omnichannel Layer — Chatwoot
Chatwoot is deployed self-hosted on **Proxmox** (VM or container) and serves as:
- The **single entry point** for all 4 channels: Website chat widget, WhatsApp Business, Instagram, Facebook Messenger
- The **live agent interface** — agents manage all conversations from one dashboard
- The **human/bot router** — client always has the choice at conversation start

Chatwoot's native **Agent Bot API** receives webhooks from all channels and forwards them to the chatbot backend. The bot responds via the same API.

### 3.2 Conversation Routing Logic
```
Client starts conversation
        │
        ▼
  Choice presented:
  [Chat with Bot] or [Talk to a Human]
        │
   ┌────┴────┐
   │         │
  Bot      Human
   │         │
   │    Agent online? ──Yes──► Live handoff in Chatwoot
   │         │
   │         No
   │         │
   │         ▼
   │    Async ticket created
   │    (agent follows up)
   ▼
Bot handles conversation
(can escalate at any point)
```

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────┐
│                  CHATWOOT                          │
│  (self-hosted on Proxmox)                          │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Channels: WA │ IG │ FB Messenger │ Website  │  │
│  └──────────────────────┬───────────────────────┘  │
│                         │                          │
│  ┌──────────┐    ┌──────▼────────┐                 │
│  │  Human   │◄───│  Conversation │                 │
│  │  Agents  │    │   Router      │                 │
│  └──────────┘    └──────┬────────┘                 │
└─────────────────────────┼──────────────────────────┘
                          │ webhook (Agent Bot API)
┌─────────────────────────▼──────────────────────────┐
│               CHATBOT BACKEND                      │
│                                                    │
│   LLM (GPT-4o — Phase 1)                          │
│   RAG Pipeline                                     │
│   WooCommerce MCP                                  │
│   Session & confidence logger                      │
└────────────┬──────────────────┬────────────────────┘
             │                  │
    ┌────────▼──────┐  ┌────────▼────────┐
    │   Vector DB   │  │   PostgreSQL    │
    │  (Qdrant or   │  │  conversations  │
    │   pgvector)   │  │  sessions, logs │
    └───────────────┘  └────────────────-┘
                                │
                       ┌────────▼────────┐
                       │   WooCommerce   │
                       │  (MCP bridge)   │
                       └─────────────────┘
```

---

## 5. RAG Pipeline

### 5.1 Knowledge Sources

| Source | Format | Indexing Method | Update Frequency |
|---|---|---|---|
| Policies, processes, how-to guides | PDF, Google Docs, Word (.docx) | Hierarchical chunking → embeddings → Vector DB | Rare (a few times/year) |
| Product descriptions, specs, features, compatibility | WooCommerce (via webhook) | Chunked → embeddings → Vector DB (per-store namespace) | On description/spec change only |
| Product price, stock availability | WooCommerce | Queried **live** via WooCommerce MCP | Real-time, never indexed |

**Product data is split across two layers:**
- **Stable product data** (descriptions, specs, compatibility, use cases) → Vector DB for semantic search
- **Volatile product data** (price, stock, SKU) → always fetched live from WooCommerce MCP

This enables semantic queries like *"compatible INGCO 20V battery tools"* while ensuring prices and availability are never stale.

### 5.2 Document Ingestion Pipeline

```
Source documents (PDF / Google Docs / Word)
            │
            ▼
      Parse & extract text
   (pypdf / Google Docs API /
    python-docx)
            │
            ▼
   Hierarchical chunking
   ┌─────────────────────────────┐
   │  Level 1: Section summary   │  ← short, ~100 tokens
   │  Level 2: Full section text │  ← detailed, ~400–600 tokens
   └─────────────────────────────┘
            │
            ▼
   Embed both levels
   (multilingual-e5-large)
            │
            ▼
   Store in Vector DB
   with metadata:
   { source, section, level,
     language: "fr", last_updated }
            │
            ▼
   Re-index triggered manually
   when a document is updated
```

**Why hierarchical chunking?**
Document volume is small (policies and guides), so the extra storage cost is negligible. Retrieval first matches at the section level, then expands to the full section text — giving precise context without losing surrounding detail.

### 5.3 Cross-lingual Retrieval (Darija → French)

Client messages arrive in Darija. Documents are in French. The multilingual embedding model maps both into the same vector space — a Darija query like *"chneya politique retour?"* retrieves the French return policy chunk correctly without any translation step.

**Model**: `multilingual-e5-large` (handles Arabic, French, Darija in one index)

### 5.4 Undocumented Knowledge Strategy

30% of operational knowledge is currently undocumented (lives in agents' heads or informal channels). Two-track approach:

| Track | Method | Timeline |
|---|---|---|
| **Fine-tuning** | Meta conversation history already captures agents' best answers — model learns from these directly | Phase 2 |
| **Progressive documentation** | In production, when the bot fails on a topic, flag it → agent documents the correct answer → re-index | Ongoing from launch |

No big upfront documentation effort. The knowledge base grows organically from real production gaps.

### 5.5 Product Update Strategy

WooCommerce sends a webhook on every product change. The backend inspects which fields changed:

```
WooCommerce product updated
          │
          ▼  (webhook)
Backend receives { product_id, store_id, changed_fields }
          │
          ├── price / stock changed only?
          │        └──► Ignore — not indexed, fetched live always
          │
          └── description / specs / features changed?
                   └──► Re-embed that product only
                        Update Vector DB store namespace
                        (single product, completes in seconds)
```

No scheduled re-index jobs. No full catalog re-index. Changes to descriptions/specs propagate instantly via webhook; price/stock are always live.

### 5.6 Two-Layer Product Query Flow

```
Client: "3andi bsahtek ta9fa INGCO 20V, chna mnajem nzid 3liha?"

Step 1 → Vector DB (store namespace)
         semantic search: "compatible INGCO 20V battery"
         → drill X, grinder Y, jigsaw Z

Step 2 → WooCommerce MCP (live)
         get_product(drill X)   → 189 TND ✓ in stock
         get_product(grinder Y) → 220 TND ✗ out of stock
         get_product(jigsaw Z)  → 165 TND ✓ in stock

Step 3 → LLM combines both layers:
         "Les outils compatibles disponibles:
          perceuse X (189 TND) et scie sauteuse Z (165 TND).
          La meuleuse Y est actuellement en rupture de stock."
```

### 5.7 Inference Flow

```
Client message (Darija / Arabic / French)
      │
      ▼
Embed message (multilingual-e5-large)
      │
      ├── Search Vector DB (shared namespace) ──► policy/process chunks
      ├── Search Vector DB (store namespace)  ──► product spec chunks
      ├── WooCommerce MCP (live)              ──► price, stock, order data
      │
      ▼
LLM prompt:
  [system prompt + store identity]
  [retrieved policy chunks]
  [retrieved product spec chunks]
  [live product/order data from MCP]
  [conversation history (last N turns)]
  [client message]
      │
      ▼
LLM generates response in client's language (Darija / Arabic / French)
```

---

## 6. WooCommerce MCP

### 6.1 Available Tools (v1)

| Tool | Description | Type |
|---|---|---|
| `search_products` | Find products by name, category, keyword | Read |
| `get_product` | Get specs, price, stock of a specific product | Read |
| `get_order` | Look up an order by ID or client info | Read |
| `track_order` | Get shipping/delivery status of an order | Read |
| `create_draft_order` | Create a pending order on behalf of client | Write |

> More tools will be added in future iterations as new capabilities are identified.

### 6.2 Order Creation Flow
```
Client: "I want to order 2x Product X"
   → Bot confirms product details and quantity with client
   → Bot calls create_draft_order
   → Order created in WooCommerce with status: "Pending Review"
   → Human agent reviews and confirms in CRM
   → Client receives confirmation
```

Draft orders are **never auto-confirmed**. All write actions land in Pending status for human validation.

---

## 7. Language & LLM Strategy

### 7.1 The Challenge
Clients write in a mix of Standard Arabic, French, and Tunisian Darija (often in the same message). Agents always mirror the client's language. The bot must do the same.

### 7.2 Phase 1 — API LLM (Launch)
- **Model**: GPT-4o
- Best available Darija understanding out of the box
- Instructed via system prompt to always mirror the client's language:

```
You are a customer support assistant for [Company].
Always respond in the exact same language and register the client uses.
If the client writes in Tunisian Darija, respond in Tunisian Darija.
If the client mixes French and Arabic, mirror that mix exactly.
Never switch languages unless the client does first.
```

### 7.3 Phase 2 — Fine-tuning (Parallel Workstream)
While Phase 1 runs in production, prepare the training dataset:

```
Meta conversation history (extracted via Graph API)
            │
            ▼
      Clean & filter
  (remove system messages, auto-replies,
   media-only messages, PII)
            │
            ▼
   Structure as training examples         ← ⚠️ SEE DECISION POINT BELOW
            │
            ▼
  Fine-tune Qwen 2.5 3B
  using QLoRA / PEFT
            │
            ▼
  Evaluate against GPT-4o baseline
  → swap only if quality matches
```

Phase 2 has **no deadline**. GPT-4o remains the production model until the fine-tuned model demonstrably matches its quality.

### 7.4 Fine-tuning Dataset — Two Sources

The training dataset combines two complementary data sources, each teaching a different capability:

| Source | Teaches | Format |
|---|---|---|
| **Meta conversation history** (client ↔ agent) | Conversational style, Darija register, client-facing tone | Windowed context pairs |
| **Mattermost threads** (agent ↔ superior) | Business decisions, exception handling, escalation logic | Question + authoritative answer pairs |

#### Source 1 — Meta Conversation History

- **Extraction**: Facebook Graph API (Page Access Token → `/{page-id}/conversations` → `/{conversation-id}/messages`)
- **Format**: Native UTF-8 JSON — Darija text is clean out of the box
- **Volume**: Tens of thousands of conversations
- **Quality**: High — agents follow consistent guidelines and scripts
- **Language**: Mostly Tunisian Darija, with French and Standard Arabic secondary

```json
{
  "source": "meta",
  "input": [
    {"role": "user",      "content": "salam, ma wasletsh commande mte3i"},
    {"role": "assistant", "content": "Ahla, 3tini numéro commande bech nchouflek"}
  ],
  "output": "Commande 1042 fi route, tousel ghodwa inchallah"
}
```

#### Source 2 — Mattermost Internal Channels

- **Extraction**: Mattermost REST API (channel history export per channel)
- **Channels**: Complaints, Orders, Technical Questions (and others)
- **Thread structure**: Short — agent asks, superior answers in 1–2 messages
- **Language**: Darija + French (internal informal register)

```json
{
  "source": "mattermost",
  "channel": "complaints",
  "input": "client yqoul commande waslet mksora, chna5ou?",
  "output": "9oulo yeb3et soura, w n3awdoulo commande jadida free of charge"
}
```

**Filtering rules for Mattermost**:
- Keep: agent question + superior answer threads
- Drop: one-liners with no actionable content, image/link-only messages, off-topic chatter, unresolved threads

#### Combined Pipeline

```
Meta Graph API                    Mattermost API
(client ↔ agent)                  (agent ↔ superior)
        │                                 │
        ▼                                 ▼
Windowed context format           Extract short Q&A threads
(3–5 turns)                       Filter non-actionable content
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
           Unified training dataset
           (each example tagged: source + channel)
                       │
                       ▼
           Fine-tune with QLoRA/PEFT
           (Qwen 2.5 3B)
                       │
                       ▼
           Evaluate per source separately
           → check Meta data taught style
           → check Mattermost data taught decisions
```

**Why tag by source?** Allows separate evaluation per capability — if one source degrades the other, weights can be adjusted before the next training run.

### 7.5 Dataset Structuring — ⚠️ DECISION REQUIRED BEFORE EXECUTION

> **This decision must be finalized before starting Phase 2 implementation.**

Three options explored for Meta data, preliminary direction is **Option B**:

| Option | Description | Pro | Con |
|---|---|---|---|
| **A) Simple pairs** | Each (client msg → agent response) = 1 example, no context | Simple, max examples | No conversation context |
| **B) Windowed context** *(preliminary)* | Last 3–5 turns included as context before agent response | Realistic, handles follow-ups | More complex to build |
| **C) Full conversation** | Entire conversation = 1 training example | Maximum context | Long sequences, fewer examples |

**Why B is preferred**: Matches how the bot operates in production (recent history available but not full conversation). Teaches the model to resolve references like "el prix mte3o" (its price) correctly.

**Before executing**: Validate by sampling 20–30 real conversations from the export to check typical conversation length, turn count, and context patterns. The optimal window size (3, 5, or 7 turns) should be decided empirically from this sample.

---

## 8. Data Layer

### 8.1 Vector DB (Qdrant)
- Collections per store (`shared`, `ingcoofficiel`, `technotools`, ...)
- Stores hierarchical document chunks + product spec embeddings
- Queried at inference time for RAG retrieval
- Updated via webhook on product/document change

### 8.2 PostgreSQL
- Conversation logs (all channels, all sessions)
- User/session profiles
- Bot decisions and confidence scores per message
- Draft orders created by bot

**Why log confidence scores?** Low-confidence bot responses are automatically flagged. When building the Phase 2 fine-tuning dataset, flagged conversations are prioritized for human review — no manual hunting through thousands of logs.

---

## 9. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Response latency | < 2s end-to-end |
| Intent recognition accuracy | > 80% on standard scenarios |
| Automation rate | > 70% without human intervention |
| Availability | 24/7 |
| Scalability | Horizontal scaling on Proxmox |
| Data privacy | INPDP compliant (Loi 2004-63), PII masked before LLM API boundary |

---

## 10. Security & Compliance

### Regulatory Context
Customers are Tunisia-based → **Loi organique 2004-63** (INPDP). All infrastructure self-hosted on Proxmox — data residency handled natively. Main exposure: OpenAI API (Phase 1) sends conversation content to US servers.

### PII Masking — LLM Boundary

Before any message reaches GPT-4o, a local masking layer strips personal identifiers:

```
Raw client message
  → PII Detection (local, on Proxmox)
  → Masked message → GPT-4o API
  → LLM never sees real PII

Raw + masked versions stored in PostgreSQL (encrypted at rest)
```

| Data type | Method | Masked as |
|---|---|---|
| Names (Arabic/French) | NER (CAMeL Tools or spaCy fr) | `[NOM]` |
| Phone numbers | Regex (Tunisian: 2x, 5x, 9x xxxxxxx) | `[TELEPHONE]` |
| Email addresses | Regex | `[EMAIL]` |
| Physical addresses | NER | `[ADRESSE]` |
| National ID (CIN) | Regex | `[CIN]` |
| Order numbers | **Not masked** — needed for WooCommerce MCP | — |
| Product names / prices | **Not masked** — not PII | — |

### Data Storage
- PostgreSQL: AES-256 encryption at rest, bound to localhost
- Qdrant: bound to localhost, no external port
- Retention: **12 months auto-purge** (configurable per store)

### API Keys & Secrets
All in environment variables, never in code or git:
`OPENAI_API_KEY`, `WOOCOMMERCE_CONSUMER_KEY_{store_id}`, `WOOCOMMERCE_CONSUMER_SECRET_{store_id}`, `CHATWOOT_API_TOKEN`, `QDRANT_API_KEY`, `RUNPOD_API_TOKEN`

### WooCommerce MCP Write Safety
- `create_draft_order` always sets status = `pending` — never auto-confirms
- `delete_order`, `delete_product`, `create_refund` never exposed to the bot
- Every tool validates `store_id` before API call
- Max 10 WooCommerce API calls per conversation turn

### Infrastructure Security

| Component | Measure |
|---|---|
| Proxmox VMs/containers | SSH key auth only |
| Public exposure | HTTPS only via Nginx reverse proxy |
| RunPod LLM endpoint | Bearer token + HTTPS, token in env var |
| Chatwoot | Role-based access: Agent / Supervisor / Admin |

### INPDP Compliance Checklist

| Requirement | How addressed |
|---|---|
| Declare processing to INPDP | Register chatbot data processing activity |
| Inform clients data is collected | Notice in chat widget welcome message |
| Right of access | Agent exports conversation on request |
| Right of deletion | Admin purges from PostgreSQL |
| Data security | AES-256 + TLS + PII masking |
| Retention limit | 12-month auto-purge |
| International transfer safeguard | PII masking before OpenAI API boundary |

---

## 9b. Multi-Store Architecture

### Store Topology

| Store | Brand scope | WooCommerce instance |
|---|---|---|
| `ingcoofficiel.tn` | INGCO brand only | Dedicated |
| `technotools.tn` | INGCO + other brands | Dedicated |
| *(future stores)* | TBD | Dedicated per store |

All stores share the same owner, same agent team, and same internal policies. Product catalogs differ per store.

### Architecture Principle: One Bot, N Stores

```
Chatwoot inbox per store
(ingcoofficiel inbox, technotools inbox, ...)
          │
          │  webhook carries store_id
          ▼
┌─────────────────────────────────────┐
│         CHATBOT BACKEND             │
│                                     │
│  store_id = "ingcoofficiel"         │
│       │                             │
│       ├── WooCommerce MCP ──► ingcoofficiel.tn API
│       │                             │
│       ├── Vector DB ──► namespace: "ingcoofficiel"
│       │                  (store-specific products)
│       │                             │
│       ├── Vector DB ──► namespace: "shared"
│       │                  (policies, processes — all stores)
│       │                             │
│       └── System prompt ──► "Tu es l'assistant de Ingco Officiel..."
│                                     │
└─────────────────────────────────────┘
```

### Vector DB Namespacing

| Namespace | Content | Shared? |
|---|---|---|
| `shared` | Return policy, shipping rules, how-to guides | All stores |
| `ingcoofficiel` | INGCO product specs, descriptions | ingcoofficiel only |
| `technotools` | INGCO + other brand products | technotools only |
| *(future store)* | Store-specific products | That store only |

Adding a new store = create a new Chatwoot inbox + new WooCommerce MCP connection + new Vector DB namespace. **No code changes needed.**

### WooCommerce MCP — Multi-store

Each store has its own WooCommerce REST API credentials. The MCP layer maps `store_id` to the correct API endpoint:

```
search_products(query, store_id="technotools")
  → hits technotools.tn WooCommerce API

create_draft_order(items, store_id="ingcoofficiel")
  → creates order on ingcoofficiel.tn WooCommerce
```

### One Model, All Stores

The fine-tuned model (Qwen 2.5 3B) is trained on data from all stores combined — the `store_id` is included in the training context so the model learns store-specific behavior naturally. No per-store models needed.

---

## 10b. Evaluation Framework (Phase 2 — Model Swap Decision)

### Goal
Determine when the fine-tuned model (Qwen 2.5 3B) is good enough to replace GPT-4o in production.

### Test Set
**150 curated test cases** (50 per dimension), covering all 3 languages (Darija, French, Standard Arabic). Each case has a reference answer (best known agent response).

| Dimension | What it tests | Example |
|---|---|---|
| **Language quality** | Correct language mirroring, natural Darija register | Client writes Darija → response must be natural Darija |
| **Business accuracy** | Correct policy, product info, order status | "chneya politique retour?" → must match actual return policy |
| **Decision-making** | Escalation triggers, refusals, order confirmation flow | Frustrated client → must trigger human handoff |

---

### Stage 1 — Automated Evaluation (fast filter)

| Metric | Dimension | Tool | Pass Threshold |
|---|---|---|---|
| Language detection match | Language quality | `langdetect` | > 90% correct language |
| Semantic similarity vs reference | Business accuracy | `multilingual-e5-large` cosine similarity | > 0.80 average |
| Decision classification accuracy | Decision-making | Rule-based + LLM judge | > 85% correct decisions |

**If any threshold fails → stop, fix, retrain. Do not proceed to Stage 2.**

---

### Stage 2 — Human (Agent) Evaluation

50 blind side-by-side comparisons (GPT-4o vs fine-tuned, randomized order, no labels):

```
Message client: "salam, 3andi commande mech waslet"

Réponse A: "Ahla, 3tini numéro commande bech..."
Réponse B: "Bonjour, pouvez-vous me donner..."

Laquelle est meilleure? [A] [B] [Égales] [Les deux mauvaises]
```

**Pass condition**: Fine-tuned model wins or ties on ≥ 70% of comparisons.

---

### Go / No-Go Decision

```
Stage 1 passes? ──No──► Fix & retrain
      │
     Yes
      │
Stage 2 passes? ──No──► Fix & retrain
      │
     Yes
      │
Deploy fine-tuned model on RunPod EU Serverless
Keep GPT-4o as instant fallback for 2 weeks
Monitor: confidence scores, escalation rate, response time
      │
No regression after 2 weeks?
      │
     Yes → Retire GPT-4o ✓
```

---

## 10c. Confidence Scoring & Active Learning

### Confidence Signal Sources

Three signals combined into a single score (0.0 – 1.0):

| Signal | Source | What it measures |
|---|---|---|
| **RAG retrieval score** | Vector DB cosine similarity | Does the knowledge base have a grounded answer? |
| **LLM self-assessment** | Structured output field (`confidence: 0.0–1.0`) | Does the model feel grounded in its response? |
| **Decision classifier** | Lightweight rule-based check | Is this a business decision requiring human judgment? |

### Threshold Logic

```
Combined confidence score
│
├── > 0.75      ──► Answer normally, no flag
│
├── 0.50–0.75   ──► LOW CONFIDENCE
│       ├── Give answer
│       ├── Signal uncertainty to client
│       ├── Offer escalation button
│       └── Flag → Chatwoot agent inbox (real-time)
│
└── < 0.50      ──► VERY LOW CONFIDENCE
        ├── Business decision? ──► Full escalation to agent
        └── Language/unclear?  ──► Ask client to rephrase
```

### Client-facing Message (Low Confidence)

```
[Darija example]
"Ma3ndich l9ina 100% bach njawbek b dabt,
 haka ndhenn: [response]

 Theb tchouf ma3 wahd min team mte3na?"
 [Oui, connectez-moi] [Non, merci]
```

Transparent, non-alarming — gives the answer AND the choice. Clicking "connect me" triggers immediate Chatwoot handoff with full context.

### Internal Flagging Flow

```
Low-confidence response generated
          │
          ▼
Stored in PostgreSQL:
{ conversation_id, message, response,
  confidence_score, rag_score, timestamp }
          │
          ├──► Chatwoot "Review" inbox (real-time)
          │    Agent sees: client message + bot response + score
          │    Agent can: correct response / mark good / escalate
          │
          └──► Weekly batch export → supervisor review
               Supervisor labels correct answer for each flagged case
```

### Active Learning Loop

```
Flagged cases + agent/supervisor corrections
          │
          ▼
Stored as new training examples
{ input, output, source: "active_learning" }
          │
          ▼
Accumulate until batch threshold (~200 new examples)
          │
          ▼
Trigger fine-tuning run (QLoRA on current model)
          │
          ▼
Evaluate (Stage 1 + Stage 2 framework)
          │
          ▼
Deploy improved model if passes ✓
```

The model improves continuously from its own production mistakes — not just the initial dataset.

---

## 10d. Dashboard & Analytics

### Principle
Chatwoot covers ~80% of dashboard needs natively with zero custom development. The remaining 20% (AI-specific metrics) is served by Metabase connected to PostgreSQL — self-hosted on Proxmox (~512MB RAM, lightweight).

---

### Layer 1 — Chatwoot Native Reports

| Report | Key Metrics | Audience |
|---|---|---|
| **Bot Reports** | Bot conversations, handoff rate, resolution rate, handoff count over time | Owner + Supervisor |
| **CSAT Reports** | Satisfaction score, response rate, emoji scale distribution | Owner |
| **Agent Reports** | First response time, resolution time, CSAT per agent, resolution count | Supervisor |
| **Inbox Reports** | Per-channel metrics — one inbox per store = automatic per-store breakdown | Owner |
| **Overview (live)** | Real-time open conversations, agent availability, unattended conversations | Supervisor |
| **SLA Reports** | SLA hit rate, SLA misses, missed SLA logs with timestamps | Supervisor |

All reports are filterable by date range, agent, inbox, and team. Exportable to CSV/Excel. Accessible via Chatwoot Reports API for programmatic access.

---

### Layer 2 — Metabase on PostgreSQL (AI-specific metrics)

Metabase connects directly to the chatbot's PostgreSQL database and provides visual dashboards for metrics Chatwoot cannot see.

**Owner dashboard (Metabase):**

| Metric | What it answers |
|---|---|
| WooCommerce draft orders created by bot | How many sales did the bot directly initiate? |
| Bot confidence score distribution over time | Is the model getting more or less certain? |
| % conversations handled without any escalation | True automation rate |
| Per-store bot performance comparison | Which store has the strongest bot coverage? |

**Supervisor dashboard (Metabase):**

| Metric | What it answers |
|---|---|
| Active learning queue size | How many flagged cases are pending review? |
| Flagged cases resolved vs pending | Is the review backlog growing or shrinking? |
| RAG retrieval score trends | Are the right document chunks being returned? |
| Low-confidence response rate by topic | Which topics need more training data? |

---

### Infrastructure

```
Proxmox Chatbot Stack (LXC container)
├── Chatwoot        ← operational reports (agents, CSAT, bot, SLA, inboxes)
├── Chatbot backend
├── PostgreSQL      ← source for AI metrics
├── Qdrant
└── Metabase        ← AI metrics dashboard (~512MB RAM, connects to PostgreSQL)
```

---

## 10e. Tech Stack

### Full Stack Summary

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Native AI/ML ecosystem — no wrappers needed |
| **Web framework** | FastAPI | Async, webhook-ready, auto-generates OpenAPI docs |
| **RAG framework** | LlamaIndex | Purpose-built for RAG; clean hierarchical chunking + multi-store namespace support |
| **LLM client (Phase 1)** | OpenAI SDK (`gpt-4o`) | Best Darija handling, fast to integrate |
| **LLM inference (Phase 2)** | Qwen 2.5 3B on RunPod EU Serverless | Fine-tuned, QLoRA, ~$28/month |
| **Embeddings** | `multilingual-e5-large` (sentence-transformers) | Handles Darija/Arabic/French in one index |
| **Vector DB** | Qdrant | Purpose-built, native collections per store, scales cleanly |
| **WooCommerce MCP** | Custom Python (MCP Python SDK) | Multi-store `store_id` routing, restricted toolset, draft-order-only write policy |
| **Document parsing** | pypdf + python-docx + Google Docs API | Covers PDF, Word, Google Docs — all 3 source formats |
| **Conversation DB** | PostgreSQL + SQLAlchemy | Structured logs, confidence scores, active learning queue |
| **Deployment** | Docker Compose | Consistent with Chatwoot, runs in Proxmox LXC container |

---

### Service Architecture (Docker Compose)

```
chatbot-stack/ (LXC container on Proxmox)
│
├── docker-compose.yml
│   ├── api          FastAPI app (chatbot backend)
│   ├── qdrant       Vector DB
│   ├── postgres     Conversation logs + confidence scores
│   ├── metabase     Analytics dashboard
│   └── chatwoot*    (* or separate VM — see chatwoot-prd.md)
```

### WooCommerce MCP — Custom Python Server

Built with the official MCP Python SDK. Exposes only the tools the bot is authorized to use:

```python
# Exposed tools (v1)
search_products(query, store_id)       # Read
get_product(product_id, store_id)      # Read
list_product_variations(product_id, store_id)  # Read
get_order(order_id, store_id)          # Read
track_order(order_id, store_id)        # Read
create_draft_order(items, store_id)    # Write — creates Pending order only
```

Dangerous WooCommerce endpoints (`delete_order`, `delete_product`, `create_refund`, etc.) are **never exposed** to the bot. Each tool enforces `store_id` routing to the correct WooCommerce API credentials.

Reference: [Automattic WooCommerce MCP](https://github.com/Automattic/ai-experiments/blob/trunk/mcp/woo/README.md) — used as reference for tool structure, reimplemented in Python.

---

## 11. Advanced Features (Future Iterations)

| Feature | Description |
|---|---|
| **RLHF** | Agents rate bot responses in Chatwoot → feedback loop into fine-tuning |
| **Active learning** | Auto-flag low-confidence responses for dataset curation |
| **Predictive analytics** | Forecast customer needs from conversation patterns |
| **Multimodal** | Analyze product images sent by clients |
| **More MCP tools** | Extend WooCommerce capabilities, connect other business systems |
| **MCP Protocol** | Connect AI to additional external data sources in real time |

---

## 11b. Infrastructure & Cost Planning

### Proxmox Resource Allocation

```
Proxmox (32GB RAM / 16 cores)
│
├── VM: Odoo           (~6GB RAM, 2 cores)   [existing]
├── VM: Plesk          (~4GB RAM, 2 cores)   [existing]
├── VM: Mattermost     (~3GB RAM, 2 cores)   [existing]
│
└── CT: Chatbot Stack  (~11GB RAM, 6 cores)  [new — LXC container]
    ├── Chatwoot           (~3GB RAM)
    ├── Chatbot backend    (~1GB RAM)
    ├── PostgreSQL         (~2GB RAM)
    ├── Qdrant             (~2GB RAM)
    └── Metabase           (~512MB RAM)
```

LLM inference is **never run on Proxmox** — no GPU available, CPU inference of a 3B model would be 30–60s per response, unacceptable for a customer chatbot.

### GPU Resources (Fine-tuning & Experimentation)

- **2× RTX 3050 laptop (4GB VRAM each)** — available for local fine-tuning experiments (QLoRA on 1B–3B models) and testing before cloud deployment
- **Cloud GPU (RunPod EU)** — for production inference (Phase 2) and larger fine-tuning runs

### LLM Hosting Cost Scenarios

> Traffic baseline: ~200 clients/day, ~1,000 queries/day, ~30,000 queries/month

| Scenario | Provider | Hardware | Monthly Cost | Recommended? |
|---|---|---|---|---|
| **A — Serverless GPU** | RunPod EU (Serverless) | RTX 4000 (16GB VRAM) | ~$16–28/month | ✅ **Yes — current volume** |
| **B — Dedicated VPS GPU** | RunPod EU (Pod) | RTX 3090/4090 (24GB) | ~$158–250/month | When volume > 500 clients/day |
| **C — Enterprise dedicated** | OVHcloud (France) | Quadro RTX 5000 (16GB) | ~$432/month | When volume > 3,000 clients/day |

**Scenario A detail** (recommended):
- Billed per second of active generation only ($0.00016/s)
- Active generation: ~90,000s/month = ~$14.40
- Storage + overhead: ~$2/month
- **With min replicas = 1 (no cold starts)**: +~$12/month idle = **~$28/month total**
- Scales automatically during flash sales or traffic spikes

**Scale triggers:**
- Migrate A → B when daily active clients exceed 500 continuously
- Migrate B → C when daily active clients exceed 3,000 continuously

### Phase 1 vs Phase 2 Cost Comparison

| Phase | LLM | Monthly Cost | Notes |
|---|---|---|---|
| **Phase 1** | GPT-4o API | ~$160/month | ~40M input tokens + ~6M output tokens at current volume |
| **Phase 2** | Fine-tuned 3B on RunPod Serverless | ~$28/month | Same volume, own model |
| **Saving** | | **~$132/month** | Phase 2 pays for itself in ~3–4 months |

### Fine-tuned Model Target

- **Model family**: Qwen 2.5 (3B) — strongest small-model Arabic/French coverage, runs on 4GB VRAM in 4-bit quantization
- **Fine-tuning method**: QLoRA (fits on RTX 3050 laptop for initial experiments)
- **Production target**: RunPod EU Serverless, 16GB VRAM tier

---

## 12. Development Phases

| Phase | Scope |
|---|---|
| **Phase 1** | Chatwoot on Proxmox + all channel connections + chatbot backend + RAG + GPT-4o + WooCommerce MCP — hosted cost ~$160/month |
| **Phase 2** | Extract & clean Meta + Mattermost dataset → fine-tune Qwen 2.5 3B with QLoRA → evaluate vs GPT-4o → deploy on RunPod EU Serverless (~$28/month) |
| **Phase 3** | Advanced features, active learning, expanded MCP tools, predictive analytics |

---

## 13. Deliverables

1. This final plan document
2. Chatwoot instance deployed on Proxmox
3. Chatbot backend (RAG + LLM + WooCommerce MCP)
4. Vector DB with indexed knowledge base
5. PostgreSQL schema and conversation logging
6. WooCommerce MCP tool set
7. Fine-tuning dataset and pipeline (Phase 2)
8. Evaluation report (Phase 2)

---

## Decisions Log

| Topic | Decision | Rationale |
|---|---|---|
| Channels | Website, WhatsApp, Instagram, Facebook Messenger | Full omnichannel as per requirements |
| Agent interface & channel layer | Chatwoot (self-hosted on Proxmox) | Handles all channels + agent UI + bot API natively |
| LLM strategy | Hybrid: GPT-4o (Phase 1) → fine-tune open-source (Phase 2) | Launch fast, improve with real data — avoids cold-start dataset problem |
| Language handling | Mirror client language (Darija/Arabic/French) via system prompt | Agents already do this; bot must match |
| RAG embeddings | Multilingual embedding model (multilingual-e5-large) | Single index for all 3 languages — handles cross-lingual retrieval (Darija query → French doc) |
| Product data | Split: stable data (descriptions, specs) → Vector DB; volatile data (price, stock) → live WooCommerce MCP | Semantic search on specs + always-fresh price/stock; webhook triggers re-embed on description change only |
| RAG chunking | Hierarchical (section summary + full section text) | Small document volume makes cost negligible; better retrieval precision |
| Undocumented knowledge (30%) | Hybrid: fine-tune on conversation history + progressive documentation from production gaps | No big upfront effort; knowledge base grows from real failures |
| Fine-tuning data sources | Meta conversation history (client ↔ agent) + Mattermost threads (agent ↔ superior) | Complementary: Meta teaches conversational style, Mattermost teaches business decisions |
| Mattermost data usage | Both fine-tuning AND RAG knowledge base source | Fills undocumented knowledge gap; short thread structure enables clean Q&A extraction |
| WooCommerce writes | Draft orders only, human confirms in CRM | Safe, auditable, no autonomous writes |
| Human escalation | Client always chooses bot or human; live if agent online, async ticket if not | Best of both, no forced bot interaction |
| Multi-store strategy | One shared bot instance, store_id per conversation | Scales to N stores with zero extra infrastructure; new store = new inbox + API credentials + Vector DB namespace |
| LLM inference hosting | RunPod EU Serverless (Scenario A) | ~$28/month at current volume; scales automatically; Phase 2 pays for itself in 3–4 months vs GPT-4o |
| Confidence scoring | 3-signal combined score (RAG retrieval + LLM self-assessment + decision classifier) | RAG score is the cleanest proxy — low similarity = model working from general knowledge |
| Low-confidence client behavior | Answer + signal uncertainty + offer escalation + internal flag | Transparent, non-alarming; client keeps control |
| Active learning review | Two-level: agents real-time (Chatwoot inbox) + supervisor weekly batch | Immediate client help + quality curation for dataset |
| Security / PII | PII masking layer before OpenAI API boundary (NER + regex, local) | LLM never sees real names, phones, addresses; INPDP safeguard for international transfer |
| Tech stack | Python + FastAPI + LlamaIndex + Qdrant + OpenAI SDK + custom WooCommerce MCP (Python) | Full AI/ML ecosystem compatibility, no cross-language friction |
| Dashboard strategy | Chatwoot native reports (operational) + Metabase on PostgreSQL (AI metrics) | Chatwoot covers agents/CSAT/bot/SLA; Metabase covers confidence scores, active learning queue, bot-influenced orders |
| Evaluation approach | Two-stage: automated metrics (Stage 1) → agent blind comparison (Stage 2) | Fast technical filter + real business quality validation |
| Model swap trigger | Stage 1 thresholds + 70% agent preference + 2-week production monitoring | No-regression safety net before retiring GPT-4o |
| Fine-tuned model target | Qwen 2.5 3B (QLoRA) | Best small-model Arabic/French/Darija coverage; fits 4GB VRAM for experiments |
| Vector DB | Qdrant | Purpose-built, native collections per store, scales cleanly as stores grow |
| Conversation DB | PostgreSQL | Structured logs + confidence scores for dataset curation |

---

## Open Questions

- [x] ~~Vector DB?~~ → **Qdrant** (dedicated, collections per store)
- [x] ~~Backend language?~~ → **Python + FastAPI + LlamaIndex + Qdrant**
- [x] ~~Cloud vs fully on-premise?~~ → **Hybrid**: services on Proxmox, LLM inference on RunPod EU Serverless
- [ ] How many documents in the initial knowledge base? (sizing the Vector DB)
- [ ] ⚠️ **BLOCKER before Phase 2**: Fine-tuning dataset structure — validate Option B (windowed context) by sampling real conversations; decide optimal window size (3, 5, or 7 turns)

Metabase integration for the Chatbot stack

1) Start the stack (from infra/):

```powershell
# from infra/ directory
docker compose up -d postgres redis qdrant chatwoot chatwoot-worker chatbot-backend metabase
```

2) Open Metabase at http://localhost:3001 and create the initial admin user (or use env METABASE_ADMIN_EMAIL/PASSWORD and the API script to log in).

3) To auto-register the Postgres DB and create the dashboard, run (requires `requests`):

```powershell
python metabase/create_metabase_resources.py --email admin@example.com --password metabase_admin_password --pg-pass "your_postgres_password"
```

4) Manual steps (if automation fails):
- In Metabase UI -> Admin -> Databases -> Add database: Postgres, host `postgres`, port `5432`, db `chatwoot_db`, user `chatwoot_user`, password from `.env`.
- Create new Questions -> Native query -> paste SQL from `sql/` files.
- Create Dashboard -> Add questions.

Notes & Caveats
- The `escalation_rate.sql` assumes a `conversation_labels` table with `conversation_id` and `title`; adapt if your schema stores labels differently.
- The `active_learning_flags` table is created by the backend active learning service and holds low-confidence events.
- The automation script uses Metabase REST API v0. Please check Metabase version compatibility if the endpoints change.

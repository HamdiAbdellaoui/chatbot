# Migrations (reference schemas)

These `.sql` files are **not executed automatically** by any tooling — there is
no migration runner wired into this project. They exist purely as a versioned,
auditable reference for the schemas the backend creates itself at runtime.

Each backend service that persists data to PostgreSQL applies its own DDL on
first use with `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`
(idempotent, safe to run repeatedly):

- `0001_active_learning_flags.sql` — `app/services/active_learning_service.py`
- `0002_conversation_logs.sql` — `app/services/conversation_log_service.py`

If you change a table's columns or indexes, update the DDL in the
corresponding service file first, then copy the exact DDL here so this
directory stays an accurate mirror of what the code actually creates.

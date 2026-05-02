# Chatwoot database setup (Docker Compose)

## 2.2 Run Chatwoot DB prepare / migrations

Your service name is `chatwoot` (not `rails`). Use these commands from the `infra/` directory.

### Option A: Official prepare task (with trace)

See exactly what the task does:

```powershell
docker compose run --rm chatwoot bundle exec rails db:chatwoot_prepare --trace
```

- You should see: `Invoke db:chatwoot_prepare` → `Execute db:migrate` → `Execute db:_dump` → "Loading Installation config". That means the DB and migrations ran; **"Loading Installation config" with no migration lines usually means "no pending migrations" (already up to date).**
- If the DB was never created, run Option B first (`db:create` then `db:migrate` then `db:seed`).

### Option B: Step-by-step (if prepare fails or you need to seed)

Run each step separately:

```powershell
# 1. Create database (if not exists)
docker compose run --rm chatwoot bundle exec rails db:create

# 2. Run migrations
docker compose run --rm chatwoot bundle exec rails db:migrate

# 3. Load seeds (optional; Chatwoot often uses in-app setup wizard instead)
docker compose run --rm chatwoot bundle exec rails db:seed
```

**Note:** If `db:create` says "Database 'chatwoot_db' already exists" and `db:migrate` only prints "Loading Installation config", the database is already set up and migrations are current.

### Option C: Full reset (only if you want a clean DB)

**Warning:** This deletes all Chatwoot data.

```powershell
docker compose run --rm chatwoot bundle exec rails db:reset
```

---

## Verify the database

List Chatwoot tables (should see many, e.g. `accounts`, `conversations`, `inboxes`):

```powershell
docker compose exec postgres psql -U chatwoot_user -d chatwoot_db -c "\dt"
```

Check that Chatwoot can connect:

```powershell
docker compose exec chatwoot bundle exec rails runner "puts ActiveRecord::Base.connection.tables.count"
```

---

## After DB setup

Restart Chatwoot so it picks up the DB state:

```powershell
docker compose restart chatwoot chatwoot-worker
```

Then open http://localhost:3000 and complete the installation wizard (create first account, etc.).

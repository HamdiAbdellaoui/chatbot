-- Escalation rate: share of outbound bot turns flagged escalated = true, per day and store.
-- Source: conversation_logs (written by chatbot-backend/app/services/conversation_log_service.py).
-- Note: escalations triggered by an explicit customer request (keywords) are answered before
-- the RAG step and are not written to conversation_logs; only low-confidence and order-request
-- escalations are counted here.
SELECT
  date_trunc('day', created_at) AS day,
  coalesce(store_key, 'unknown') AS store_key,
  count(*) FILTER (WHERE escalated) AS escalated_turns,
  count(*) AS outbound_turns,
  round(100.0 * count(*) FILTER (WHERE escalated) / NULLIF(count(*), 0), 2) AS percent_escalated
FROM conversation_logs
WHERE direction = 'out'
GROUP BY 1, 2
ORDER BY 1 DESC, 2
LIMIT 500;

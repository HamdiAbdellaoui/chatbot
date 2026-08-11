-- Escalation rate: percent of conversations labeled with the human handoff label per day
-- Adjust label join if your Chatwoot schema uses a different table for labels.
SELECT
  date_trunc('day', c.created_at) AS day,
  sum(CASE WHEN EXISTS (
    SELECT 1 FROM conversation_labels cl WHERE cl.conversation_id = c.id AND cl.title = 'human_handoff'
  ) THEN 1 ELSE 0 END) AS escalated_count,
  count(*) AS total_conversations,
  (sum(CASE WHEN EXISTS (
    SELECT 1 FROM conversation_labels cl WHERE cl.conversation_id = c.id AND cl.title = 'human_handoff'
  ) THEN 1 ELSE 0 END)::float / NULLIF(count(*),0)) * 100.0 AS percent_escalated
FROM conversations c
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;

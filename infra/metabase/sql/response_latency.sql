-- Response latency: average time (seconds) between conversation creation and first reply per day
SELECT
  date_trunc('day', c.created_at) AS day,
  avg(EXTRACT(EPOCH FROM (c.first_reply_created_at - c.created_at))) AS avg_first_reply_seconds,
  count(*) AS conversations
FROM conversations c
WHERE c.first_reply_created_at IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;

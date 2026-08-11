-- Low-confidence flags: recent active_learning_flags events
SELECT
  date_trunc('day', created_at) AS day,
  count(*) AS flags_count,
  avg(top_score) AS avg_top_score
FROM active_learning_flags
WHERE event_type = 'low_confidence'
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;

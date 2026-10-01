-- Bot response latency percentiles (LLM generation time, milliseconds) per day.
-- Source: conversation_logs.latency_ms on outbound turns.
SELECT
  date_trunc('day', created_at) AS day,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS latency_p50_ms,
  percentile_cont(0.9) WITHIN GROUP (ORDER BY latency_ms) AS latency_p90_ms,
  count(*) AS outbound_turns
FROM conversation_logs
WHERE direction = 'out'
  AND latency_ms IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;

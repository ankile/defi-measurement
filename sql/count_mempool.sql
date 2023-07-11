SELECT 
  DATE_TRUNC('hour', first_seen) as hour,
  COUNT(hash) as transaction_count
FROM
	mempool_transactions
GROUP BY
	DATE_TRUNC('hour', first_seen)
ORDER BY
	DATE_TRUNC('hour', first_seen);

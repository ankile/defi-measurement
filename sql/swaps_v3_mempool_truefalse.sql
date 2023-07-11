SELECT 
  block_number,
  address,
  SUM(CASE WHEN from_mempool = true THEN 1 ELSE 0 END) AS mempool_true, 
  SUM(CASE WHEN from_mempool = false THEN 1 ELSE 0 END) AS mempool_false,
  COUNT(*) AS total
FROM
  swaps
WHERE
	block_number >= 17552205
GROUP BY 
  block_number, address
ORDER BY 
  total DESC
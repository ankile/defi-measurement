SELECT 
  block_number, 
  SUM(CASE WHEN from_mempool = true THEN 1 ELSE 0 END) AS mempool_true, 
  SUM(CASE WHEN from_mempool = false THEN 1 ELSE 0 END) AS mempool_false
FROM 
  swaps_v2
GROUP BY 
  block_number
ORDER BY 
  block_number ASC
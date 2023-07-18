SELECT 
  COUNT(DISTINCT(block_number))
FROM
  swaps
WHERE
	block_number >= 17552205
SELECT 
    EXTRACT(EPOCH FROM (swaps.block_timestamp - date_trunc('second', mempool_transactions.first_seen))) AS diff_in_seconds,
    COUNT(*) AS frequency
FROM 
    swaps
INNER JOIN 
    mempool_transactions 
    ON swaps.transaction_hash = mempool_transactions.hash
GROUP BY
    diff_in_seconds
HAVING 
    EXTRACT(EPOCH FROM (swaps.block_timestamp - date_trunc('second', mempool_transactions.first_seen))) <= 35
ORDER BY
    diff_in_seconds ASC;

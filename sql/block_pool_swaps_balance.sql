SELECT
   BLOCK_NUMBER,
   ADDRESS,
   CNT,
   NEGATIVE_STARTS,
   NON_NEGATIVE_STARTS,
   CAST(ABS(NEGATIVE_STARTS - NON_NEGATIVE_STARTS) AS FLOAT) / CNT AS BALANCE 
FROM
   (
      SELECT
         BLOCK_NUMBER,
         ADDRESS,
         COUNT(*) AS CNT,
         COUNT(
         CASE
            WHEN
               AMOUNT0 LIKE '-%' 
            THEN
               1 
         END
) AS NEGATIVE_STARTS, COUNT(
         CASE
            WHEN
               AMOUNT0 NOT LIKE '-%' 
            THEN
               1 
         END
) AS NON_NEGATIVE_STARTS 
      FROM
         SWAPS 
      GROUP BY
         BLOCK_NUMBER, ADDRESS
   )
   AS COUNTS 
WHERE
   CNT > 4 
ORDER BY
   BALANCE ASC, CNT DESC LIMIT 100
// Import PrismaClient
import { PrismaClient } from "@prisma/client";

type QueryResult = {
  block_number: number;
  mempool_true: number;
  mempool_false: number;
};

const prisma = new PrismaClient();

async function main() {
  const result: QueryResult[] = await prisma.$queryRaw`
    SELECT 
      block_number, 
      SUM(CASE WHEN from_mempool = true THEN 1 ELSE 0 END) AS mempool_true, 
      SUM(CASE WHEN from_mempool = false THEN 1 ELSE 0 END) AS mempool_false
    FROM 
      swaps
    GROUP BY 
      block_number
    ORDER BY 
      block_number ASC;
  `;

  console.log(result);
}

main()
  .catch((e) => {
    throw e;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

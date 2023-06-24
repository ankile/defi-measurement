/*
  Warnings:

  - You are about to drop the `mempool_transaction` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "mempool_transaction";

-- CreateTable
CREATE TABLE "MemPoolTransaction" (
    "transaction_hash" VARCHAR NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MemPoolTransaction_pkey" PRIMARY KEY ("transaction_hash")
);

-- CreateIndex
CREATE INDEX "ix_mempool_transaction_timestamp" ON "MemPoolTransaction"("timestamp");

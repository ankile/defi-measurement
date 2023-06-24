/*
  Warnings:

  - You are about to drop the `MemPoolTransaction` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "MemPoolTransaction";

-- CreateTable
CREATE TABLE "MempoolTransaction" (
    "transaction_hash" VARCHAR NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MempoolTransaction_pkey" PRIMARY KEY ("transaction_hash")
);

-- CreateIndex
CREATE INDEX "ix_mempool_transaction_timestamp" ON "MempoolTransaction"("timestamp");

-- CreateTable
CREATE TABLE "mempool_transaction" (
    "transaction_hash" VARCHAR NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "mempool_transaction_pkey" PRIMARY KEY ("transaction_hash")
);

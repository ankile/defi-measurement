-- CreateTable
CREATE TABLE "swaps" (
    "transaction_hash" VARCHAR,
    "block_timestamp" TIMESTAMP(6) NOT NULL,
    "block_number" INTEGER NOT NULL,
    "transaction_index" INTEGER NOT NULL,
    "log_index" INTEGER NOT NULL,
    "sender" VARCHAR NOT NULL,
    "recipient" VARCHAR NOT NULL,
    "amount0" VARCHAR NOT NULL,
    "amount1" VARCHAR NOT NULL,
    "sqrtPriceX96" VARCHAR NOT NULL,
    "liquidity" VARCHAR NOT NULL,
    "tick" VARCHAR NOT NULL,
    "address" VARCHAR NOT NULL,
    "from_address" VARCHAR NOT NULL,
    "to_address" VARCHAR NOT NULL,
    "from_mempool" BOOLEAN,

    CONSTRAINT "swaps_pkey" PRIMARY KEY ("block_number","transaction_index","log_index")
);

-- CreateIndex
CREATE INDEX "ix_swaps_address" ON "swaps"("address");

-- CreateIndex
CREATE INDEX "ix_swaps_block_number" ON "swaps"("block_number");

-- CreateIndex
CREATE INDEX "ix_swaps_from_address" ON "swaps"("from_address");

-- CreateIndex
CREATE INDEX "ix_swaps_recipient" ON "swaps"("recipient");

-- CreateIndex
CREATE INDEX "ix_swaps_sender" ON "swaps"("sender");

-- CreateIndex
CREATE INDEX "ix_swaps_to_address" ON "swaps"("to_address");

-- CreateIndex
CREATE INDEX "ix_swaps_transaction_hash" ON "swaps"("transaction_hash");


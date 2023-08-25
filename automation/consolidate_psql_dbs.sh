#!/bin/bash

# Load environment variables from .env file
source /home/azureuser/defi-measurement/.env

# Source and target database names
SOURCE_DB="mempool"
TARGET_DB="uniswap"

# echo "Backing up $SOURCE_DB database..."
pg_dump -h "$PGHOST" -d "$SOURCE_DB" -U "$PGUSER" -f /home/azureuser/database-dumps/${SOURCE_DB}.sql --clean --no-password --verbose

# echo "Backing up $TARGET_DB database..."
# pg_dump -h "$PGHOST" -d "$TARGET_DB" -U "$PGUSER" -f /home/azureuser/database-dumps/${TARGET_DB}.sql --no-password --verbose


echo "Migrating $SOURCE_DB database to $TARGET_DB..."
# Connect to the target database
psql -h "$PGHOST" -U "$PGUSER" -d $TARGET_DB <<EOF

-- Set verbosity to verbose for detailed output
\set VERBOSITY verbose

-- Import data from the dump file
\i /home/azureuser/database-dumps/${SOURCE_DB}.sql


EOF

echo "Migration completed successfully!"

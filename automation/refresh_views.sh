#!/bin/bash

# Load environment variables from .env file
source /home/azureuser/defi-measurement/.env

# SQL command to refresh the materialized view
SQL="REFRESH MATERIALIZED VIEW mempool_rate_per_hour;"

echo "Refreshing materialized view..."

# Execute the SQL command with psql
psql -h "$PGHOST" -d "$PGDATABASE" -U "$PGUSER" -c "$SQL" --no-password

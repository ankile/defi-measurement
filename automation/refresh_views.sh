#!/bin/bash

# PostgreSQL connection details
HOST="your_host"
DB="your_database"
USER="your_user"
PASS="your_password"

# SQL command to refresh the materialized view
SQL="REFRESH MATERIALIZED VIEW my_view;"

# Execute the SQL command with psql
export PGPASSWORD="$PASS"
psql -h "$HOST" -d "$DB" -U "$USER" -c "$SQL"
unset PGPASSWORD

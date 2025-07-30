#!/bin/bash
set -e

# Create database directory if it doesn't exist
mkdir -p /app/databases

# Run database setup if the database doesn't exist
if [ ! -f /app/databases/hospital.db ]; then
    echo "Initializing database..."
    cd /app/databases
    python db_setup.py
    echo "Database initialization complete."
fi

# Start the application
exec "$@"
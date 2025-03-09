#!/bin/bash

# Capture the current working directory
CURRENT_DIR="$(pwd)"

# Define PostgreSQL directory locations using absolute paths
PGDATA="$HOME/.learning_db"
PGSOCKETDIR="$PGDATA/pg_socket"
PGLOG="$PGDATA/postgres.log"
PGUSER="agent"
PGPASSWORD="password"
DBNAME="learning_db"

# Function to check if PostgreSQL is installed
check_postgres_installed() {
    if ! command -v initdb &>/dev/null || ! command -v pg_ctl &>/dev/null || ! command -v psql &>/dev/null; then
        echo "Error: PostgreSQL is not installed. Please install it first."
        exit 1
    fi
}

# Function to initialize the database if not already initialized
initialize_database() {
    if [ ! -d "$PGDATA" ]; then
        echo "Initializing PostgreSQL database at $PGDATA..."
        initdb -D "$PGDATA"
    fi
}

# Function to check if PostgreSQL is running
is_postgres_running() {
    pg_ctl -D "$PGDATA" status &>/dev/null
    return $? # Returns 0 if running, non-zero otherwise
}

# Function to gracefully stop PostgreSQL if running
stop_postgres() {
    if is_postgres_running; then
        echo "PostgreSQL is running. Stopping it..."
        pg_ctl -D "$PGDATA" stop -m fast
        sleep 3
    fi

    # If PostgreSQL is still running, force kill it
    if is_postgres_running; then
        echo "PostgreSQL did not stop. Forcing shutdown..."
        PID=$(head -n 1 "$PGDATA/postmaster.pid" 2>/dev/null)
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            kill -TERM "$PID"
            sleep 3
        fi
    fi

    # Final kill if necessary
    if is_postgres_running; then
        echo "Force kill failed. Trying kill -9..."
        kill -9 "$PID" 2>/dev/null
        sleep 3
    fi
}

# Function to ensure necessary directories exist
prepare_directories() {
    echo "Ensuring socket directory exists at $PGSOCKETDIR..."
    mkdir -p "$PGSOCKETDIR"
    chmod 700 "$PGSOCKETDIR" # Restrict access for security
}

# Function to start PostgreSQL
start_postgres() {
    stop_postgres # Ensure no conflicting instances
    echo "Starting PostgreSQL server..."
    pg_ctl -D "$PGDATA" -l "$PGLOG" -o " -k $PGSOCKETDIR -c listen_addresses=''" start

    echo "PostgreSQL is running locally!"
    echo "To connect, use: psql  -h $PGSOCKETDIR -d $DBNAME"
    echo "To stop the server, run: pg_ctl -D \"$PGDATA\" stop"
}

# Function to create the user if not exists
ensure_user_exists() {
    echo "Ensuring user '$PGUSER' exists..."
    psql -h "$PGSOCKETDIR" -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$PGUSER';" | grep -q 1 ||
        psql -h "$PGSOCKETDIR" -d postgres -c "CREATE USER $PGUSER WITH PASSWORD '$PGPASSWORD' SUPERUSER;"
}

# Function to create the database if not exists
ensure_database_exists() {
    echo "Ensuring database '$DBNAME' exists..."
    psql -h "$PGSOCKETDIR" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DBNAME';" | grep -q 1 ||
        psql -h "$PGSOCKETDIR" -d postgres -c "CREATE DATABASE $DBNAME OWNER $PGUSER;"
}

### MAIN SCRIPT EXECUTION ###
check_postgres_installed
initialize_database
prepare_directories
start_postgres
sleep 3 # Wait for PostgreSQL to be fully operational
ensure_user_exists
ensure_database_exists

echo "PostgreSQL setup is complete. Ready to use!"

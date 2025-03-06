#!/bin/bash

# Define variables
CURRENT_DIR=$(pwd)
mkdir -p "$CURRENT_DIR/data/alfworld_postgresql"
PGDATA="$HOME/.learning_db_postgresql"
PGSOCKETDIR="$PGDATA/pg_socket"
PGLOG="$PGDATA/postgres.log"
DBNAME="learning_db"

# Check if PostgreSQL is running
is_postgres_running() {
    pg_ctl -D "$PGDATA" status &>/dev/null
    return $? # Returns 0 if running, non-zero otherwise
}

# Start PostgreSQL server
start_postgres() {
    echo "Initializing PostgreSQL database at $PGDATA..."
    mkdir -p "$PGDATA"
    if [ ! -f "$PGDATA/PG_VERSION" ]; then
        initdb -D "$PGDATA"
    fi
    mkdir -p "$PGSOCKETDIR"
    chmod 700 "$PGSOCKETDIR"
    pg_ctl -D "$PGDATA" -l "$PGLOG" -o " -k $PGSOCKETDIR -c listen_addresses=''" start
    echo "PostgreSQL is running locally!"
    echo "To connect, use: psql -h $PGSOCKETDIR -d $DBNAME"
    echo "To stop the server, run: pg_ctl -D \"$PGDATA\" stop"
}

# Stop PostgreSQL server and dump the database
stop_postgres() {
    if [[ -n "$LOG_NAME" ]]; then
        pg_dump -h "$PGSOCKETDIR" -d "$DBNAME" -f "$CURRENT_DIR/data/alfworld_postgresql/$LOG_NAME.sql"
    else
        echo "LOG_NAME not set, skipping database dump."
    fi
    pg_ctl -D "$PGDATA" stop -m fast
}

# Cleanup function for handling interruptions
cleanup() {
    echo "Terminating processes..."
    if [[ -n "$PYTHON_PID" ]]; then
        kill $PYTHON_PID 2>/dev/null
        wait $PYTHON_PID 2>/dev/null
    fi
    echo "Stopping the PostgreSQL server..."
    stop_postgres
    exit 1
}

# Set trap for Ctrl+C
trap cleanup SIGINT

# Start PostgreSQL if not already running
if ! is_postgres_running; then
    start_postgres
fi
sleep 5

# Main loop for training with different parallel settings
for parallel in 1 4 16; do
    LOG_NAME="expel_train_postgresql_parallel_${parallel}"

    # Reset database to base state
    dropdb -h "$PGSOCKETDIR" "$DBNAME" 2>/dev/null || true
    createdb -h "$PGSOCKETDIR" "$DBNAME"
    psql -h "$PGSOCKETDIR" -d "$DBNAME" -f "$CURRENT_DIR/data/alfworld_postgresql/learning_base.sql"
    sleep 5 # Allow time for initialization

    # Run training script in the background
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_type postgresql \
        --db_path "$PGSOCKETDIR" \
        --log_name "$LOG_NAME" \
        --num_passes 1 \
        --env alfworld \
        --store_episodes \
        --num_ic 3 \
        --parallel "$parallel" \
        --num_tasks 3500 &

    # Capture and wait for Python process
    PYTHON_PID=$!
    wait "$PYTHON_PID"

    # Dump the database after training
    pg_dump -h "$PGSOCKETDIR" -d "$DBNAME" -f "$CURRENT_DIR/data/alfworld_postgresql/$LOG_NAME.sql"
done

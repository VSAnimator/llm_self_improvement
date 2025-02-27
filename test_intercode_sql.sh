# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
'''
for train_db_size in 10 15 20 40 60 100 150 200 ; do
    python data/transform_chroma.py /mnt/ssd/agent_algo_bench/data/intercode_sql_bird_backups/${train_db_size}/learning.db/learning.db /mnt/ssd/agent_algo_bench/data/intercode_sql_bird_backups/${train_db_size}/learning.db/
done
'''

for train_db_size in 200 ; do
    # Run the training script in the background
    python scripts/run_agent_v2.py --llm openai/gpt-4o --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 338 --start_task 238 --db_path "$CURRENT_DIR/data/intercode_sql_bird_backups/${train_db_size}/learning.db/learning.db" --log_name bird_gold_${train_db_size}_chroma_retry --parallel 10 --db_type sqlite --start_task 272 &

    # Capture Python process ID
    PYTHON_PID=$!

    # Wait for the Python script to finish
    wait $PYTHON_PID
done

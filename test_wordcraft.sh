# Get current directory
CURRENT_DIR=$(pwd)

# Sweep db sizes 
for db_size in 10 20 40 100 200 400 1000 2000
do
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft --num_tasks 5000 --parallel 3 --num_ic 50 --start_task 0 --log_name wordcraft_depth_2_humanic_4tries_db_size_$db_size --db_path $CURRENT_DIR/data/wordcraft/depth2_humanic_4tries_backups/$db_size/learning.db/learning.db --start_task 4000 &
    sleep 5
done

wait
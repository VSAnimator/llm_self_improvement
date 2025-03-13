# Get current directory
CURRENT_DIR=$(pwd)

# Just run rap_noplan
python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft --num_tasks 5000 --parallel 10 --num_ic 50 --start_task 0 --log_name wordcraft_depth_2_humanic_4tries --db_path $CURRENT_DIR/data/wordcraft/depth2_humanic_4tries_backups/3600/learning.db/learning.db --start_task 4004 &

wait
# Get current directory
CURRENT_DIR=$(pwd)

# Just run rap_noplan
for trial in 1 2 3 4 5;
do
    echo "Running trial $trial"
    # First copy over the directory
    cp -r $CURRENT_DIR/data/wordcraft/depth2_humanic $CURRENT_DIR/data/wordcraft/depth2_humanic_train_$trial

    # Run the script
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft --num_tasks 4000 --parallel 1 --num_ic 10 --start_task 0 --log_name wordcraft_depth_2_humanic_train_$trial --db_path $CURRENT_DIR/data/wordcraft/depth2_humanic_train_$trial/learning.db --store_episodes &

    sleep 30
done

wait
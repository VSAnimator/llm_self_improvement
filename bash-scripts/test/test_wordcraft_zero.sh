# Get current directory
CURRENT_DIR=$(pwd)

# Later we need to sweep DB sizes
# Run the script
for trial in 3 4 5;
do
    for db_size in 10 100 200 400 1000 2000 3700;
    do
        python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft_test --num_tasks 500 --parallel 4 --num_ic 10 --start_task 0 --log_name zero_${trial}_10ic_${db_size} --db_path $CURRENT_DIR/data/wordcraft/zero_${trial}_backups/${db_size}/learning.db/learning.db &
        sleep 10
    done
    wait
done

'''
# Run the script
for trial in 1 2 3 4 5;
do
    for db_size in 10 100 200 400 1000 2000 3700;
    do
        python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft_test --num_tasks 500 --parallel 5 --num_ic 30 --start_task 0 --log_name wordcraft_depth_2_humanic_test_${trial}_30ic_${db_size} --db_path $CURRENT_DIR/data/wordcraft/depth2_humanic_train_${trial}_backups/${db_size}/learning.db/learning.db &
        sleep 10
    done
    wait
done
'''

# Also have to do it on the default db
'''
for trial in 1 2 3 4 5;
do
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env wordcraft_test --num_tasks 500 --parallel 5 --num_ic 10 --start_task 0 --log_name wordcraft_depth_2_humanic_default_${trial} --db_path $CURRENT_DIR/data/wordcraft/depth2_humanic/learning.db &
    sleep 10
done
'''

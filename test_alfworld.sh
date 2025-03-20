# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
'''
for ic in 6; do
    for trial in 1 2 3 4 5; do
        for ckpt in 20 40 100 200 400 1000 1500 2000 2500 3000; do
            python scripts/run_agent_v2.py \
                --llm openai/gpt-4o-mini \
                --agent_type rap_flex \
                --db_path "$CURRENT_DIR/data/alfworld_expel_trial_${trial}_${ic}_ic_backups/${ckpt}/learning.db/learning.db" \
                --log_name trial_${trial}_${ic}ic_${ckpt} \
                --num_passes 1 \
                --env alfworld_test \
                --num_ic $ic \
                --parallel 2 \
                --num_tasks 134 &
            sleep 5
        done
        wait
    done
done
'''

for ic in 3; do
    for trial in 1 2 3 4 5; do
        for ckpt in 20 40 100 200 400 1000 1500 2000 2500 3000; do
            python scripts/run_agent_v2.py \
                --llm openai/gpt-4o-mini \
                --agent_type rap_flex \
                --db_path "$CURRENT_DIR/data/alfworld_expel_trial_${trial}_backups/${ckpt}/learning.db/learning.db" \
                --log_name trial_${trial}_${ic}ic_${ckpt} \
                --num_passes 1 \
                --env alfworld_test \
                --num_ic $ic \
                --parallel 2 \
                --num_tasks 134 &
            sleep 5
        done
        wait
    done
done
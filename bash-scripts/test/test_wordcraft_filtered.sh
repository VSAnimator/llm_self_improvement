# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in {1..5}; do
    #for ckpt in 40 100 200 400 1000 1500 2000 2500 3000 3500 4000; do
    for ckpt in 4000; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_noplan \
            --db_path "$CURRENT_DIR/data/wordcraft_filtered/wordcraft_worst_examples_${ckpt}/learning.db" \
            --log_name wordcraft_worst_${ckpt}_trial${trial}_redo \
            --num_passes 1 \
            --env wordcraft_test \
            --num_ic 10 \
            --parallel 10 \
            --num_tasks 500
        sleep 5
    done
done

'''
for ckpt in 40 100 200 400 800 1000 1500 2000 2500 3000 3500 4000; do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_noplan \
        --db_path "$CURRENT_DIR/data/wordcraft_filtered/wordcraft_worst_examples_${ckpt}/learning.db" \
        --log_name wordcraft_worst_${ckpt} \
        --num_passes 1 \
        --env wordcraft_test \
        --num_ic 10 \
        --parallel 10 \
        --num_tasks 500
    sleep 5
done
'''
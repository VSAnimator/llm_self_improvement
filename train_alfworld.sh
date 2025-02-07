# Get current directory
CURRENT_DIR=$(pwd)

# Create new database folders by copying existing one in parallel
for ic in 3 6 10; do
    cp -r "$CURRENT_DIR/data/alfworld_expel" "$CURRENT_DIR/data/alfworld_expel_${ic}ic" &
done

# Wait for all copy operations to complete
wait

# Run training script for each number of in-context examples in parallel
for ic in 3 6 10; do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/alfworld_expel_${ic}ic/learning.db" \
        --db_name expel_train_${ic}ic \
        --num_passes 1 \
        --env alfworld \
        --store_episodes \
        --num_ic $ic \
        --num_tasks 3500 &
done

# Wait for all training processes to complete
wait
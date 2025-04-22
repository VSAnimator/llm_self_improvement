# Set the current directory
CURRENT_DIR=$(pwd)

# Set the archive directory
ARCHIVE_DIR="$CURRENT_DIR/data/alfworld_archives"

# What is the archive name?
ARCHIVE_NAME="alfworld_expel_trial_dbs.tar.gz"

# Compress only the specified backup databases into a single archive
# Use quotes around each path pattern to handle wildcards properly
tar -czf "$ARCHIVE_DIR/$ARCHIVE_NAME" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/20/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/40/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/100/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/200/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/400/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/1000/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/1500/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/2000/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/2500/learning.db" \
    "$CURRENT_DIR/data/alfworld_expel_trial_"*"_6_ic_backups/3000/learning.db"

echo "Archive created at: $ARCHIVE_DIR/$ARCHIVE_NAME"
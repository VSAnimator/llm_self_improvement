# Get current directory
CURRENT_DIR=$(pwd)

# Compress all the database folders into a single archive
echo "Compressing all database folders into a single archive..."
ARCHIVE_NAME="wordcraft_depth2_humanic_all_dbs.tar.gz"
echo "Creating archive: $ARCHIVE_NAME"

# Create a list of all database paths to include in the archive
DB_PATHS=""
for trial in 1 2 3 4 5;
do
    for db_size in 10 100 200 400 1000 2000 3700;
    do
        DB_PATHS="$DB_PATHS $CURRENT_DIR/data/wordcraft/depth2_humanic_train_${trial}_backups/${db_size}"
    done
done

# Create the single archive with all database folders
tar -czf "$ARCHIVE_NAME" $DB_PATHS
echo "Archive created: $ARCHIVE_NAME"
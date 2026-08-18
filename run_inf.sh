#!/bin/bash
while true; do
    echo "Starting script..."
    python -m aleksis_scripts.download_data_tmp
    echo "Script ended. Restarting in 2 seconds..."
    sleep 2
done

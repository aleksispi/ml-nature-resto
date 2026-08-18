#!/bin/bash
while true; do
    echo "Starting script..."
    python download_data_tmp.py
    echo "Script ended. Restarting in 2 seconds..."
    sleep 2
done
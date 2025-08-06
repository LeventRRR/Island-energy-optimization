#!/bin/bash

CSV_FILE="chosen_island.csv"
MAIN_LOG="logs/main_log.log"
# --- 定义一个文件，专门记录未达到Gap目标的岛屿 ---
GAP_FAIL_LOG="logs/gap_failure_islands.csv"

> "$MAIN_LOG"  # 清空主日志文件
# --- 初始化Gap失败日志文件并写入表头 ---
echo "ID,Lat,Long,Task,FinalGap" > "$GAP_FAIL_LOG"

# --- 定义一个函数来检查日志文件中的MIPGap ---
# 参数: 1:log_file, 2:task_name, 3:island_id, 4:island_lat, 5:island_lon
check_mip_gap() {
    local log_file=$1
    local task_name=$2
    local island_id=$3
    local island_lat=$4
    local island_lon=$5
    local target_gap_percent=1.00 # 您的目标是1%，这里写成1.00用于比较

    # 使用grep查找包含关键字的行
    local gap_line=$(grep "Time limit reached. Using best solution found with gap" "$log_file")

    # 检查是否找到了这一行
    if [ -n "$gap_line" ]; then
        # 如果找到了，提取gap百分比数值。
        local final_gap=$(echo "$gap_line" | awk '{print $NF}' | tr -d '%')
        
        # 使用bc进行浮点数比较，如果 final_gap > target_gap_percent
        if (( $(echo "$final_gap > $target_gap_percent" | bc -l) )); then
            echo "GAP CHECK FAILED: Island $island_id, Task $task_name. Final Gap: ${final_gap}% > ${target_gap_percent}%" >> "$MAIN_LOG"
            # 将失败信息记录到专门的CSV文件中
            echo "$island_id,$island_lat,$island_lon,$task_name,${final_gap}%" >> "$GAP_FAIL_LOG"
        fi
    fi
}


tail -n +2 "$CSV_FILE" | while IFS=',' read -r ID Long Lat Country Island Pop Geometry Region; do
    LOG_PREFIX="logs/log_${ID}"
    TASK_SUCCESS=true

    run_task_with_retry() {
        TASK_NAME=$1
        SCRIPT_NAME=$2
        LOG_FILE=$3

        echo "$TASK_NAME started for ID: $ID at $(date +'%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
        
        # 运行并重试逻辑保持不变
        if ! python3 "$SCRIPT_NAME" --island_lat "$Lat" --island_lon "$Long" --pop "$Pop" >> "$LOG_FILE" 2>&1; then
            sleep 10
            if ! python3 "$SCRIPT_NAME" --island_lat "$Lat" --island_lon "$Long" --pop "$Pop" >> "$LOG_FILE" 2>&1; then
                echo "$TASK_NAME failed for ID: $ID at $(date +'%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
                return 1
            fi
        fi

        echo "$TASK_NAME completed for ID: $ID at $(date +'%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
        return 0
    }

    ISLAND_START_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo "Island $ID: PROCESSING STARTED at $ISLAND_START_TIME" >> "$MAIN_LOG"

    # --- 串行执行任务 ---

    # 1. 运行 disaster_2020 任务，等待其完成
    run_task_with_retry "disaster_2020" "disaster_2020.py" "${LOG_PREFIX}_disaster_2020.log"
    DISASTER_2020_EXIT=$? # 直接捕获退出码
    check_mip_gap "${LOG_PREFIX}_disaster_2020.log" "disaster_2020" "$ID" "$Lat" "$Long"
    
    # 2. 运行 disaster_2050 任务，等待其完成
    run_task_with_retry "disaster_2050" "disaster_2050.py" "${LOG_PREFIX}_disaster_2050.log"
    DISASTER_2050_EXIT=$? # 直接捕获退出码
    check_mip_gap "${LOG_PREFIX}_disaster_2050.log" "disaster_2050" "$ID" "$Lat" "$Long"

    # --- 任务执行完毕，开始统计和报告 ---
    
    ISLAND_END_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    ISLAND_START_TIMESTAMP=$(date -d "$ISLAND_START_TIME" +%s)
    ISLAND_END_TIMESTAMP=$(date -d "$ISLAND_END_TIME" +%s)
    ISLAND_DURATION=$((ISLAND_END_TIMESTAMP - ISLAND_START_TIMESTAMP))
    HOURS=$((ISLAND_DURATION / 3600))
    MINUTES=$(((ISLAND_DURATION % 3600) / 60))
    SECONDS=$((ISLAND_DURATION % 60))
    DURATION_FORMAT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)

    # 检查任务退出状态
    if [ $DISASTER_2020_EXIT -ne 0 ] || [ $DISASTER_2050_EXIT -ne 0 ]; then
        TASK_SUCCESS=false
    fi

    if [ "$TASK_SUCCESS" = true ]; then
        echo "Island $ID: ALL TASKS COMPLETED at $ISLAND_END_TIME (Duration: $DURATION_FORMAT)" >> "$MAIN_LOG"
    else
        echo "Island $ID: TASKS FAILED (2020:$DISASTER_2020_EXIT, 2050:$DISASTER_2050_EXIT) at $ISLAND_END_TIME (Duration: $DURATION_FORMAT)" >> "$MAIN_LOG"
    fi
    
    echo "" >> "$MAIN_LOG"
done
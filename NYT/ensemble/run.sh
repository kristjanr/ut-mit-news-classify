# Run `scheduler.sh` in the background and save its PID 

model=$1
chunk_size=$2
timestamp=`date +"%s"`
pid_file="scheduler_${timestamp}.pid"
log_file="scheduler_${timestamp}.log"

nohup bash scheduler.sh > $log_file 2>&1 &
echo $! > $pid_file

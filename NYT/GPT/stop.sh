# kill the nohup process to stop scheduling trainers
run="$1"
pid_file="${run}.pid"
log_file="${run}.log"
pid=`cat $pid_file`

echo "`date +"%D %T"` Stopping scheduler (PID=$pid) manually..." >> $log_file
kill -9 $pid && echo "Stopped scheduler successfully."
rm $pid_file
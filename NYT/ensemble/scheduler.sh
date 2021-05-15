# Schedule instance of `trainer.sh` jobs to SLURM as soon as previous one starts

schedule_trainer() {
  while true
  do
    # set output folder names
    timestamp=`date +"%s"`

    # add job to queue
    echo "`date +"%D %T"` Adding run ${timestamp} to queue..."
    sbatch vectorize-slurm.sh 

    # let's schedule next job
    echo "`date +"%D %T"` Run ${timestamp} started! Scheduling next run in 8h30m..."
    sleep 8h
    sleep 30m

  done
}

schedule_trainer
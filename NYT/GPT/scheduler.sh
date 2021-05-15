# Schedule instance of `trainer.sh` jobs to SLURM as soon as previous one starts

model=$1
chunk_size=$2

schedule_trainer() {
  while true
  do
    # set output folder names
    timestamp=`date +"%s"`

    # add job to queue
    echo "`date +"%D %T"` Adding run ${timestamp} with ${model} to queue..."
    sbatch vectorize-gpt2-slurm.sh $model $chunk_size

    # let's schedule next job
    echo "`date +"%D %T"` Run ${timestamp} with ${model} started! Scheduling next run in 8h30m..."
    sleep 8h
    sleep 30m

  done
}

schedule_trainer
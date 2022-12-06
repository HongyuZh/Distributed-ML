echo "begin to run docker"
for ((i = 0; i < 5; i++)) do
    docker run --name s_worker_$i -d \
        -v ../DistributedML_DOCKER/file \
        -e pattern_k=5 \
        -e worker_index=$i \
        -e num_workers=5 \
        -e batch_size=8 \
        -e epoches=50 \
        -e agg_mod=batch_100 \
        -e seed=1234 \
        scatter_reduce_docker
    echo "docker${i} is built"
done 
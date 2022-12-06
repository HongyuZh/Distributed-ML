# Distributed-ML

* First execute `divide.py` to divide the cifar-10 training set and test set. These data will be stored in file. (already executed)
* Then `cd scatter_reduce/Dockerfile`, execute the command -`docker built -t scatter_reduce_docker .`. **Notice the image name must be scatter_reduce_docker.**
* Last run the bash file `run.sh`. (You can adjust most of the parameters there)

*Some paths in file `alexnet_cifar10.py` and `worker.py` are localized, you should change them according to you own environment.*
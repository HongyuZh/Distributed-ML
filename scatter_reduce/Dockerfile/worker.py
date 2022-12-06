import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import pickle
import logging

from alexnet_cifar10 import Net, load_data

def scatter_reduce(weights, epoch, num_workers, worker_index, pattern_k, batch = 0):
    # turn the weights into a 1-d numpy array
    vector = weights[0].reshape(1,-1)
    for i in range(1,len(weights)):
        vector = np.append(vector, weights[i].reshape(1,-1))
        # vector is supposed to be a 1-d numpy array
    num_all_values = vector.size
    num_values_per_agg = num_all_values // pattern_k
    residue = num_all_values

    # write partitioned vector to the shared memory, except the chunk charged by myself
    for i in range(pattern_k):
        if i != worker_index:
            # decide which part of the weight should be put in the shared memory
            offset = (num_values_per_agg * i) + min(residue, i)
            length = num_values_per_agg + (1 if i < residue else 0)
            # indicating the chunk number and which worker it comes from
            # format of key in tmp-bucket: chunkID_epoch_batch_workerID
            key = "{}_{}_{}_{}".format(i, epoch, batch, worker_index)
            tmp_file = '/home/starfish/workspace/Distributed_DOCKER/file/tmp/' + key
            file = open(tmp_file, 'wb')
            pickle.dump(vector[offset: offset + length], file)
            file.close()

    merged_value = dict()
  
    # aggregator only
    if worker_index < pattern_k:
        my_offset = (num_values_per_agg * worker_index) + \
            min(residue, worker_index)
        my_length = num_values_per_agg + (1 if worker_index < residue else 0)
        my_chunk = vector[my_offset: my_offset + my_length]
        
        # read and aggregate the corresponding chunk
        tmp_prefix = "{}_{}_{}_".format(worker_index, epoch, batch)
        for index in range(num_workers):
            if index != worker_index:
                cur_file = '/home/starfish/workspace/Distributed_DOCKER/file/data/cifar-10-pictures-test' + \
                    tmp_prefix + str(index)
                # other workers haven't put their data in the shared memory
                while not os.path.exists(cur_file):
                    time.sleep(0.5)
                flag = True
                while flag:
                    try:
                        cur_data = pickle.load(open(cur_file, 'rb'))
                        flag = False
                    except EOFError:
                        time.sleep(0.5)
                my_chunk += cur_data
                os.remove(cur_file)
    
        # average weights
        my_chunk /= float(num_workers)
        # write the aggregated chunk back
        # key format in merged_bucket: epoch_batch_chunkID
        key = "{}_{}_{}".format(epoch, batch, worker_index)
        merged_dir = '/home/starfish/workspace/Distributed_DOCKER/file/data/cifar-10-pictures-test' + \
            str(epoch)
        try:
            os.mkdir(merged_dir)
        except FileExistsError:
            time.sleep(0.1)
        merged_file = merged_dir + '/' + key
        file = open(merged_file, 'wb')
        pickle.dump(my_chunk, file)
        file.close()
        merged_value[worker_index] = my_chunk

    # read other aggregated chunks
    merged_prefix = "{}_{}_".format(epoch, batch)
    for index in range(pattern_k):
        if index != worker_index:
            cur_file = '/home/starfish/workspace/Distributed_DOCKER/file/data/cifar-10-pictures-test' + \
                str(epoch) + '/' + merged_prefix + str(index)
            while not os.path.exists(cur_file):
                time.sleep(0.5)
            flag = True
            while flag:
                try:
                    cur_data = pickle.load(open(cur_file, 'rb'))
                    flag = False
                except EOFError:
                    time.sleep(0.5)
            merged_value[index] = cur_data

    # reconstruct the whole vector
    new_vector = merged_value[0]
    for k in range(1, pattern_k):
        new_vector = np.concatenate((new_vector, merged_value[k]))
    
    result = dict()
    index = 0
    for k in range(len(weights)):
        lens = weights[k].size
        tmp_arr = new_vector[index:index + lens].reshape(weights[k].shape)
        result[k] = tmp_arr
        index += lens
        
    return result

# use this function to ensure that when the seed remain the same,
# every training process will produce the same result.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# training setting
# these parameters are introduced through "run.sh"
pattern_k = int(os.environ['pattern_k'])  # num of aggregator
worker_index = int(os.environ['worker_index'])  # index of worker
num_workers = int(os.environ['num_workers'])  # total num of workers
batch_size = int(os.environ['batch_size'])
epoches = int(os.environ['epoches'])
agg_mod = os.environ['agg_mod']  # use "epoch" as default
seed = int(os.environ['seed'])

setup_seed(seed)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if agg_mod != 'epoch':
    splits = agg_mod.split("_")
    num_batches = int(splits[1])
else:
    num_batches = 'epoch'

num_pics = 5000 // num_workers * 10  # the cifar-10 training set has 10 classes, each contained 5000 items
num_mini_batches = num_pics // batch_size # the number of batches we need to run
num_output_m_b = num_mini_batches // 5 # 5 stands for output 5 times

net = Net()
# every worker has its own data according to worker_index.
trainloader, testloader = load_data(batch_size, worker_index)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

count_batches = 0

logging.info(f"this is worker {worker_index}")
logging.info(f"batch size is {batch_size}")
for epoch in range(epoches):  # loop over the dataset multiple times
    logging.info(f"epoch is {epoch+1}")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % num_output_m_b == num_output_m_b - 1: 
            output_loss = running_loss / num_output_m_b
            logging.info(f'[{i+1}] : {output_loss:.3f}')
            running_loss = 0.0

        count_batches += 1
        if agg_mod != 'epoch' and count_batches % num_batches == 0:
            # scatter_reduce every 'num_batches' batches
            weights = [param.data.numpy() for param in net.parameters()]
            merged_weights = scatter_reduce(
                weights, epoch+1, num_workers, worker_index, pattern_k, i+1)
            for layer_index, param in enumerate(net.parameters()):
                param.data = torch.from_numpy(merged_weights[layer_index])
    logging.info(f"total batch is {i+1}")

    if agg_mod == 'epoch':
        # scatter_reduce at the end of epoch
        # extract the parameters from the net
        weights = [param.data.numpy() for param in net.parameters()]
        # invoke the function scatter_reduce to merge the weights
        merged_weights = scatter_reduce(weights, epoch, num_workers, worker_index, pattern_k)
        for layer_index, param in enumerate(net.parameters()):
            param.data = torch.from_numpy(merged_weights[layer_index])

logging.info(f'Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

logging.info(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
logging.info(f'Finished Testing')



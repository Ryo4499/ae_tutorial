# AE tutorial

Trained AE model by cifar10.

Compared to the original implementation, we have added batchnorm and changed the compression dimension.

paper

> <https://arxiv.org/abs/2201.03898>

## FAA

Feed-forward architecture

### MNIST

```txt
epoch: 189
test_loss          0.010081866756081581
```

![test batch 0](results/datasets/mnist/faa/images/batch_0.png)
![test batch 1](results/datasets/mnist/faa/images/batch_1.png)
![test batch 2](results/datasets/mnist/faa/images/batch_2.png)
![test batch 3](results/datasets/mnist/faa/images/batch_3.png)
![test batch 4](results/datasets/mnist/faa/images/batch_4.png)
![test batch 5](results/datasets/mnist/faa/images/batch_5.png)
![test batch 6](results/datasets/mnist/faa/images/batch_6.png)
![test batch 7](results/datasets/mnist/faa/images/batch_7.png)
![test batch 8](results/datasets/mnist/faa/images/batch_8.png)
![test batch 9](results/datasets/mnist/faa/images/batch_9.png)

### CIFAR10

```txt
epoch: 48
test_loss           0.03823928162455559
```

![test batch 0](results/datasets/cifar10/faa/images/batch_0.png)
![test batch 1](results/datasets/cifar10/faa/images/batch_1.png)
![test batch 2](results/datasets/cifar10/faa/images/batch_2.png)
![test batch 3](results/datasets/cifar10/faa/images/batch_3.png)
![test batch 4](results/datasets/cifar10/faa/images/batch_4.png)
![test batch 5](results/datasets/cifar10/faa/images/batch_5.png)
![test batch 6](results/datasets/cifar10/faa/images/batch_6.png)
![test batch 7](results/datasets/cifar10/faa/images/batch_7.png)
![test batch 8](results/datasets/cifar10/faa/images/batch_8.png)
![test batch 9](results/datasets/cifar10/faa/images/batch_9.png)

## CA

Convolutional auto encoder architecture

### MNIST

```txt
epoch: 68
test_loss          4.327146598370746e-05
```

![test batch 0](results/datasets/mnist/ca/images/batch_0.png)
![test batch 1](results/datasets/mnist/ca/images/batch_1.png)
![test batch 2](results/datasets/mnist/ca/images/batch_2.png)
![test batch 3](results/datasets/mnist/ca/images/batch_3.png)
![test batch 4](results/datasets/mnist/ca/images/batch_4.png)
![test batch 5](results/datasets/mnist/ca/images/batch_5.png)
![test batch 6](results/datasets/mnist/ca/images/batch_6.png)
![test batch 7](results/datasets/mnist/ca/images/batch_7.png)
![test batch 8](results/datasets/mnist/ca/images/batch_8.png)
![test batch 9](results/datasets/mnist/ca/images/batch_9.png)

### CIFAR10

```txt
epoch:
epoch: 86
test_loss          0.001922393450513482
```

![test batch 0](results/datasets/cifar10/ca/images/batch_0.png)
![test batch 1](results/datasets/cifar10/ca/images/batch_1.png)
![test batch 2](results/datasets/cifar10/ca/images/batch_2.png)
![test batch 3](results/datasets/cifar10/ca/images/batch_3.png)
![test batch 4](results/datasets/cifar10/ca/images/batch_4.png)
![test batch 5](results/datasets/cifar10/ca/images/batch_5.png)
![test batch 6](results/datasets/cifar10/ca/images/batch_6.png)
![test batch 7](results/datasets/cifar10/ca/images/batch_7.png)
![test batch 8](results/datasets/cifar10/ca/images/batch_8.png)
![test batch 9](results/datasets/cifar10/ca/images/batch_9.png)

## Usage

If you test this script, please remove below in Dockerfile.

```Dockerfile
WORKDIR /root
COPY setup_vim/ ./setup_vim/
RUN ./setup_vim/neovim/scripts/install_dependencies_root.sh && \
    ./setup_vim/neovim/scripts/install_dependencies_user.sh
```

```sh
git clone $REPO_URL
cd ae_tutorial
cp .env.sample .env
# Specify your environments
vi .env
docker compose build
docker compose up -d
docker compose exec app sh
./train.sh
docker compose down
```

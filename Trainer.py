from network import SimpleNetwork


class Trainer:
    def __init__(self, model, train_loader, cuda_avail, optimizer, loss_function):
        self.model = model
        self.train_loader = train_loader
        self.cuda_avail = cuda_avail
        self.optimizer = optimizer
        self.loss_function = loss_function

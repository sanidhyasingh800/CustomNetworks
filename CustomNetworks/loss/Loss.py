class Loss:
    def __init__(self):
        pass

    # find the loss for a single or batch of training examples 
    # expected and output must be matrices with single example per row
    def calculate_loss(self, expected, output):
        pass


    # find the derivative of the loss for a single or batch of training examples 
    # expected and output must be matrices with single example per row
    # used in backpropagation of the output layer
    # derivative taken w.r.t to the output term
    def loss_derivative(self, expected, output):
        pass

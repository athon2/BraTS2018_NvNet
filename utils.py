import pickle
import csv
import torch

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


#def calculate_accuracy(outputs, targets):
#    batch_size = targets.size(0)
#
#    _, pred = outputs.topk(1, 1, True)
#    pred = pred.t()
#    correct = pred.eq(targets.view(1, -1))
#    n_correct_elems = correct.float().sum().data[0]
#
#    return n_correct_elems / batch_size
def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)

def dice_coefficient(outputs, targets, eps=1e-8):
    batch_size = targets.size(0)
    y_pred = outputs[:,0,:,:,:]
    y_truth = targets[:,0,:,:,:]

    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union 
    
    return dice / batch_size
    
def normalize_data(data, mean, std):
    pass
    
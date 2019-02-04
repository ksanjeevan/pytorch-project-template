
import torch
from tqdm import tqdm
class ClassificationEvaluator(object):

    def __init__(self, data_loader, model):

        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)


    def evaluate(self, metrics):
        with torch.no_grad():
            total_metrics = torch.zeros(len(metrics))
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                batch_size = data.shape[0]
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(output, target) * batch_size

            size = len(self.data_loader.dataset)
            ret = {met.__name__ : total_metrics[i].item() / size for i, met in enumerate(metrics)}
            return ret

from fastai.learner import *


class RegressionLearner(Learner):

    def __init__(self, data, models, **kwargs):
        super().__init__(data, models,  **kwargs)
        self.crit = models.loss
        self.metrics = [self.accuracy]

    def accuracy(x, y):
        mask = y[0]
        return accuracy_thresh(0.5)(x, mask)
    # def mean_iou(self, x, y):
    #     x = to_np(x)
    #     y = to_np(y)
    #     return batch_databowl_metric(x, y)

    @property
    def model(self):
        return self.models

    @property
    def data(self):
        return self.data_

    def get_layer_groups(self):
        return [self.model] #TODO: moze to jest problemem, nie powinno to byc na modelu?


    def predict_batch(self, images):
        images = self.data.to_gpu(images)
        #with torch.no_grad(): # pytorch 4.0
        return self.data.from_gpu(predict_batch(self.model, images))

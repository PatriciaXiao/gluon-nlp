import mxnet as mx

# https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/logistic_regression_explained.html

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Trainer
from mxnet.gluon.data import DataLoader, ArrayDataset

mx.random.seed(12345)  # Added for reproducibility

train_data_size = 1000
val_data_size = 100
batch_size = 10

X = mx.nd.array([[2.85225797, 0.],
     [2.93938112, 0.],
     [2.32653689, 0.],
     [3.20769596, 0.],
     [2.50110292, 0.],
     [2.74856758, 0.],
     [2.34768701, 0.],
     [2.63620615, 0.],
     [3.10402179, 0.], 
     [3.25548005, 0.], 
     [3.2242589 , 0.], 
     [2.8941226 , 0.], 
     [2.77163219, 0.], 
     [3.13367057, 0.], 
     [3.0664885 , 0.], 
     [2.33734465, 0.], 
     [3.18818521, 0.], 
     [2.46791553, 0.], 
     [2.54324341, 0.], 
     [3.55956101, 0.], 
     [2.30429435, 0.], 
     [2.33923841, 0.], 
     [3.0537591 , 0.], 
     [2.34042096, 0.], 
     [3.35983133, 0.], 
     [3.19644165, 0.], 
     [2.57492638, 0.], 
     [2.55594516, 0.], 
     [2.2650044 , 0.], 
     [2.8443327 , 0.], 
     [3.26606679, 0.], 
     [2.37013745, 0.], 
     [3.29041243, 0.], 
     [3.20673275, 0.], 
     [2.14603281, 0.], 
     [3.10267878, 0.], 
     [2.60819101, 0.], 
     [3.30423427, 0.], 
     [2.5299921 , 0.], 
     [3.34893465, 0.], 
     [2.50838995, 0.], 
     [2.78786588, 0.], 
     [2.73144102, 0.], 
     [3.67248535, 0.], 
     [2.80196905, 0.], 
     [2.77713823, 0.], 
     [2.76427293, 0.], 
     [2.31659579, 0.], 
     [3.67917919, 0.], 
     [3.06422377, 0.], 
     [2.55387783, 0.], 
     [1.94631386, 0.], 
     [3.3632102 , 0.], 
     [2.3587718 , 0.], 
     [2.61273599, 0.], 
     [1.93489075, 0.], 
     [2.60092878, 0.], 
     [2.72776151, 0.], 
     [2.19706678, 0.], 
     [3.17768216, 0.], 
     [3.15465236, 0.], 
     [3.13678408, 0.], 
     [2.97425628, 0.], 
     [2.94958949, 0.], 
     [2.19988561, 0.], 
     [2.52655935, 0.], 
     [3.41875219, 0.], 
     [2.74899101, 0.], 
     [4.58227777, 0.], 
     [2.49357128, 0.], 
     [2.51663327, 0.], 
     [2.37622428, 0.], 
     [3.24528027, 0.], 
     [2.54408574, 0.], 
     [3.01287293, 0.], 
     [3.10996294, 0.], 
     [3.0704751 , 0.], 
     [2.56283879, 0.], 
     [3.53959537, 0.], 
     [2.54891539, 0.], 
     [3.11256337, 0.], 
     [3.09313536, 0.], 
     [3.73695636, 0.], 
     [3.09887934, 0.], 
     [3.01350999, 0.], 
     [3.89441562, 0.], 
     [2.85534954, 0.], 
     [4.46888733, 0.], 
     [2.46555543, 0.], 
     [2.62967563, 0.], 
     [3.36556172, 0.], 
     [2.47922707, 0.], 
     [2.44500971, 0.], 
     [2.61866164, 0.], 
     [3.2263937 , 0.], 
     [2.26351023, 0.], 
     [2.60946178, 0.], 
     [3.29584217, 0.], 
     [3.2323966 , 0.], 
     [2.24986458, 0.], 
     [4.56159258, 0.], 
     [3.291116  , 0.], 
     [2.62658668, 0.], 
     [3.41776824, 0.], 
     [2.49941754, 0.], 
     [2.30417967, 0.], 
     [3.92098093, 0.], 
     [3.45178175, 0.], 
     [3.02981663, 0.], 
     [2.7961781 , 0.], 
     [2.19799995, 0.], 
     [2.54483414, 0.], 
     [1.70642424, 0.], 
     [2.17384601, 0.], 
     [4.4414773 , 0.], 
     [2.62963414, 0.], 
     [3.63789129, 0.], 
     [2.82681441, 0.], 
     [3.55169106, 0.], 
     [2.60926104, 0.], 
     [3.28557038, 0.], 
     [2.24772096, 0.], 
     [2.70114446, 0.], 
     [2.57273388, 0.], 
     [2.79165363, 0.], 
     [2.84786463, 0.], 
     [3.22600627, 0.]])

y = mx.nd.array([1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1.,
         1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1.,
         0., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1.,
         0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
         1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1.,
         0., 1., 0., 0., 0., 1., 0.])

val_x = mx.nd.array([[4., 0.], [2., 0.], [-2., 0.]])
val_ground_truth_class = mx.nd.array([0., 1., 1.])

train_dataset = ArrayDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ArrayDataset(val_x, val_ground_truth_class)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

classifier = nn.HybridSequential()
with classifier.name_scope():
    classifier.add(nn.Dense(units=10, activation='relu'))  # input layer
    classifier.add(nn.Dense(units=10, activation='relu'))   # inner layer 1
    classifier.add(nn.Dense(units=10, activation='relu'))   # inner layer 2
    classifier.add(nn.Dense(units=1))   # output layer: notice, it must have only 1 neuron
classifier.initialize(mx.init.Xavier())


loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = Trainer(params=classifier.collect_params(), optimizer='sgd',
                  optimizer_params={'learning_rate': 0.1})
accuracy = mx.metric.Accuracy()
f1 = mx.metric.F1()

def train_model():
    cumulative_train_loss = 0

    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            # Do forward pass on a batch of training data
            output = classifier(data)

            # Calculate loss for the training data batch
            loss_result = loss(output, label)

        # Calculate gradients
        loss_result.backward()

        # Update parameters of the network
        trainer.step(batch_size)

        # sum losses of every batch
        cumulative_train_loss += nd.sum(loss_result).asscalar()

    return cumulative_train_loss

def validate_model(threshold):
    cumulative_val_loss = 0

    #for i, (val_data, val_ground_truth_class) in enumerate(val_dataloader):
    if len(val_x) < 10:
        val_data = val_x
        # Do forward pass on a batch of validation data
        output = classifier(val_data)

        # Similar to cumulative training loss, calculate cumulative validation loss
        cumulative_val_loss += nd.sum(loss(output, val_ground_truth_class)).asscalar()

        # getting prediction as a sigmoid
        prediction = classifier(val_data).sigmoid()

        # Converting neuron outputs to classes
        predicted_classes = mx.nd.ceil(prediction - threshold)

        # Update validation accuracy
        accuracy.update(val_ground_truth_class, predicted_classes.reshape(-1))

        # calculate probabilities of belonging to different classes. F1 metric works only with this notation
        prediction = prediction.reshape(-1)
        probabilities = mx.nd.stack(1 - prediction, prediction, axis=1)

        f1.update(val_ground_truth_class, probabilities)

        # print(val_ground_truth_class, predicted_classes)
        # input()

    return cumulative_val_loss

epochs = 10
threshold = 0.5

for e in range(epochs):
    avg_train_loss = train_model() / train_data_size
    avg_val_loss = validate_model(threshold) / val_data_size

    print("Epoch: %s, Training loss: %.2f, Validation loss: %.2f, Validation accuracy: %.2f, F1 score: %.2f" %
          (e, avg_train_loss, avg_val_loss, accuracy.get()[1], f1.get()[1]))

    # we reset accuracy, so the new epoch's accuracy would be calculated from the blank state
    accuracy.reset()


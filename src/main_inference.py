import pdb
import os
import numpy as np
from sklearn.metrics import roc_curve

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn import functional

from cls_TIS_dataset import TISDataset
from cls_TIS_model import TISRoverPlus


def simple_test_run(model, data_loader, device_id):
    """
    Args:
        model (Pytorch model): NN model to run
        data_loader (Pytorch DataLoader): Data loader to get predictions on
        file_path (string): Path to file that contains exported results
        file_name (string): File  name to  export results
        run_mode (string): Name of the run, for printing etc
    """
    print('Test starts')
    # Model is wrapped around DataParallel
    model = model.eval()
    model = model.cuda(device_id)
    # Output files
    data_size = data_loader.batch_size * len(data_loader)
    correctly_classified = 0

    correct_label_list = []
    pred_label_list = []
    pred_conf_list = []
    pred_logit_list = []
    for data_index, (images, labels, _) in enumerate(data_loader):
        # --- Forward pass begins -- #
        # Send data to gpu
        if torch.cuda.is_available():
            images = images.cuda(device_id)
        # Convert images and labels to variable
        images = images
        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        # If cuda, Get data to cpu
        if torch.cuda.is_available():
            outputs = outputs.cpu()
        prob_outputs = functional.softmax(outputs, dim=1)
        # Get predictions
        prediction_confidence, predictions = torch.max(prob_outputs, 1)
        prediction_logit, _ = torch.max(outputs, 1)
        # --- Forward pass ends -- #

        # --- File export output format begins --- #
        correctly_classified += sum(np.where(predictions.numpy() == labels.numpy(), 1, 0))
        # Convert outs to list
        predicted_as = list(predictions.numpy())
        true_labels = list(labels.numpy())
        prediction_confidence = list(prediction_confidence.numpy())
        prediction_logit = list(prediction_logit.numpy())
        # Extend the big list
        correct_label_list.extend(true_labels)
        pred_label_list.extend(predicted_as)
        pred_conf_list.extend(prediction_confidence)
        pred_logit_list.extend(prediction_logit)
        # --- File export output format ends --- #

    # Calculate accuracy
    acc = "{0:.4f}".format(correctly_classified / data_size)
    print('Test ends')
    return acc, correct_label_list, pred_label_list, pred_conf_list, pred_logit_list


def extract_class_accuracy(correct_labels, pred_labels):
    """
        Get per-class accuracy based on the predictions
    """
    class_acc = [0 for x in set(correct_labels)]
    class_tot_samples = [0 for x in set(correct_labels)]

    for true, pred in zip(correct_labels, pred_labels):
        class_tot_samples[true] = class_tot_samples[true] + 1
        if true == pred:
            class_acc[true] = class_acc[true] + 1
    for index, (class_tot, class_corr) in enumerate(zip(class_tot_samples, class_acc)):
        class_acc[index] = class_acc[index] / class_tot_samples[index]
    return class_acc    # 0 for non-tis, 1 for tis


def load_pytorch_model(folder_path, file_name):
    """
    Args:
        folder_path (string): Path to file that results will be read from
        file_name (string): Name of the file that results will be read from

    returns:
        model (Pytorch model): loaded model
    """
    file_with_path = os.path.join(folder_path, file_name)
    model = torch.load(file_with_path)
    return model


def calc_fpr_at_tpr80(true_labels, pred_labels, pred_conf):
    y_score = []
    for pred, conf in zip(pred_labels, pred_conf):
        if pred == 0:
            conf = 1 - conf
        y_score.append(conf)

    fpr, tpr, thresholds = roc_curve(true_labels, y_score, pos_label=1)

    for f, t in zip(fpr, tpr):
        if t >= 0.8:
            return f
    return 0


if __name__ == "__main__":
    print('Started')
    # Load the data
    ts_dataset = TISDataset(['../data/beta_globin.pos'])
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=64, shuffle=False)

    # Which GPU to use
    DEVICE_ID = 0
    # Path to the model
    model_path = '../model/tis_rover_plus.pth'

    # Initialize the model
    model = TISRoverPlus()
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    print('Loaded:', model_path)

    model = model.eval()
    # Test run
    test_acc, true_labels, pred_labels, pred_conf, pred_logit = simple_test_run(model, ts_loader, DEVICE_ID)
    print('Accuracy:', test_acc)
    # Get TPR and FPR
    acc_per_class = extract_class_accuracy(true_labels, pred_labels)
    print('TPR:', acc_per_class[0], 'FPR:', acc_per_class[1])
    # Get FPR at sens80
    fpr_out = calc_fpr_at_tpr80(true_labels, pred_labels, pred_conf)
    print('FPR at tpr80:', fpr_out)

    print('Finished')

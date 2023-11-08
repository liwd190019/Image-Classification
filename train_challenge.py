"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
# from model.challenge import Challenge
from model.challenge import *
from train_common import *
from challenge_target import *
# import challenge_source as chas
from model.challenge_source import *
from utils import config
import utils
import copy

def freeze_layers(model, size=0):
    """
    Args:
        model (challenge): a model built in challenge.py
        size (int, optional): the size of layers to be frozen. Defaults to 0.
    Due to the time limit, I only implemented 
    1. half of the conv layers are freezed: size=1
    2. the whole conv layers are freezed: size=2
    """
    if size <= 0:
        return
    num = size
    for param in model.parameters():
        if num == 0:
            break
        param.requires_grad = False
        num -= 0.5

def train(tr_loader, va_loader, te_loader, model, model_name, num_layers=0):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge target model with type", num_layers, "frozen")
    model, start_epoch, stats = restore_checkpoint(
        model, model_name
    )

    axes = utils.make_training_plot("Challenge Target Training")

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, include_test=True
    )

    # initial val loss for early stopping
    global_min_loss = stats[0][1]

    # TODO: Define patience for early stopping. Replace "None" with the patience value.
    patience = 12
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, include_test=True
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        # TODO: Implement early stopping
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        #
        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    utils.save_challenge_training_plot()
    utils.hold_training_plot()


def main():
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target", batch_size=config("challenge.batch_size"), augment=True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
        )
    
    do_transfer_learning = input("use transfer learning?y/n\n")
    if do_transfer_learning == 'n':
        use_which_model = input("use which model? resnet/original\n")
        if use_which_model == 'original':
            model = Challenge2()
        if use_which_model == 'resnet':
            model = getResNet18()
        
        # TODO: Define loss function and optimizer. Replace "None" with the appropriate definitions.
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.009)

        # Attempts to restore the latest checkpoint if exists
        print("Loading challenge...")
        model, start_epoch, stats = restore_checkpoint(
            model, config("challenge.checkpoint")
        )

        axes = utils.make_training_plot()

        # Evaluate the randomly initialized model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, include_test=True
        )

        # initial val loss for early stopping
        global_min_loss = stats[0][1]

        # TODO: Define patience for early stopping. Replace "None" with the patience value.
        patience = 5
        curr_count_to_patience = 0

        # Loop over the entire dataset multiple times
        epoch = start_epoch
        while curr_count_to_patience < patience:
            # Train model
            train_epoch(tr_loader, model, criterion, optimizer)

            # Evaluate model
            evaluate_epoch(
                axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, include_test=True,
            )

            # Save model parameters
            save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

            # TODO: Implement early stopping
            curr_count_to_patience, global_min_loss = early_stopping(
                stats, curr_count_to_patience, global_min_loss
            )
            #
            epoch += 1
        print("Finished Training")
        # Save figure and keep plot open
        utils.save_challenge_training_plot()
        utils.hold_training_plot()
    else:
        # freeze_none = getResNet18_target()
        freeze_none = getResNet18_source()
        print("Loading source ...")
        freeze_none, _, _ = restore_checkpoint(
            freeze_none, config("challenge_source.checkpoint"), force=True, pretrain=True
        )
        
        freeze_whole = copy.deepcopy(freeze_none)
        freeze_layers(freeze_whole, 10)
        
        # modify the last layer:
        num_class = 2
        freeze_none.fc = torch.nn.Linear(freeze_none.fc.in_features, num_class)
        freeze_whole.fc = torch.nn.Linear(freeze_whole.fc.in_features, num_class)
        # freeze_layers.fc = torch.nn.Linear(freeze_layers.fc.in_features, num_classes)

        
        # train(tr_loader, va_loader, te_loader, freeze_none, "./checkpoints/challenge_target0/", 0)
        train(tr_loader, va_loader, te_loader, freeze_whole, "./checkpoints/challenge_target1/", 1)
        
    # Model
    # model = Challenge()
    # resnet_8class, _, _ = restore_checkpoint(
    #     freeze_none, config("challenge_source.checkpoint"), force=True, pretrain=True
    # )
    
    # resnet_8class = nn.Sequential(*list(resnet_8class.children())[:-1])
    
    # add own classifier
    # num_class = 2
    # classifier = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(512, num_classes)  # 512 is the number of features in the ResNet's output
    # )

    # Combine pre-trained ResNet and new classifier
    # model = nn.Sequential(resnet_8class, classifier)
    
    
    
    

    


if __name__ == "__main__":
    main()

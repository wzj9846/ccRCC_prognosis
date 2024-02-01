import os
import numpy as np

import torch

from tqdm import tqdm
from lifelines.utils import concordance_index

# import EarlyStopping
from models.earlystop import EarlyStopping

def train_survival(model, data_in, model_dir, criterion, optimizer, scheduler, max_epochs, patience=20, device=torch.device("cuda:0")):
    '''train function
    params:
    data_in: a tuple of (train_loader, test_loader)
    model_dir: the path of saving params and result
    '''
    train_loader, test_loader = data_in

    best_metric = -1
    best_metric_train = -1
    best_metric_epoch = -1

    save_loss_train = []
    save_loss_test = []
    save_cindex_train = []
    save_cindex_test = []

    save_lr_history = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, delta=0, path=os.path.join(model_dir,"early_stop_model.pth"), verbose=True)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")

        model.train()
        train_epoch_loss = 0
        train_step = 0
        train_c_index = 0
        
        batch_train_risk_pred = []
        batch_train_true_e = []
        batch_train_true_y = []

        for batch_data in tqdm(train_loader):
            train_step += 1
            X = batch_data["image"].to(device)
            e = batch_data["event"].to(device)
            y = batch_data["time"].to(device)

            risk_pred = model(X)

            # print(risk_pred)

            train_loss = criterion(risk_pred, y, e)

            # print(f"loss{train_loss}")
            
            # Calculate the c index in all train samples
            batch_train_risk_pred.extend(risk_pred.detach().cpu().numpy())
            batch_train_true_e.extend(e.cpu().numpy())
            batch_train_true_y.extend(y.cpu().numpy())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
       
        train_epoch_loss /= train_step
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        train_c_index = concordance_index(batch_train_true_y, batch_train_risk_pred, batch_train_true_e)
        save_cindex_train.append(train_c_index)
        np.save(os.path.join(model_dir, 'cindex_train.npy'), save_cindex_train)

        if train_c_index > best_metric_train:
            best_metric_train = train_c_index
            torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_train_model.pth"))

        print('Epoch_loss: {:4f} Epoch_cindex: {:4f}'.format(train_epoch_loss, train_c_index))

        model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            test_c_index = 0
            test_step = 0

            batch_test_risk_pred = []
            batch_test_true_e = []
            batch_test_true_y = []

            for test_data in tqdm(test_loader):

                test_step += 1

                X = test_data["image"].to(device)
                e = test_data["event"].to(device)
                y = test_data["time"].to(device)

                risk_pred = model(X)

                test_loss = criterion(risk_pred, y, e)
                test_epoch_loss += test_loss.item()

                # Calculate the c index in all test samples
                batch_test_risk_pred.extend(risk_pred.cpu().numpy())
                batch_test_true_e.extend(e.cpu().numpy())
                batch_test_true_y.extend(y.cpu().numpy())
                

            test_epoch_loss /= test_step
            save_loss_test.append(test_epoch_loss)
            np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

            test_c_index = concordance_index(batch_test_true_y, batch_test_risk_pred, batch_test_true_e)
            save_cindex_test.append(test_c_index)
            np.save(os.path.join(model_dir, 'cindex_test.npy'), save_cindex_test)

            print('Epoch_loss: {:4f} Epoch_cindex: {:4f}'.format(test_epoch_loss, test_c_index))

            # save the best c-index model in valisation
            if test_c_index > best_metric:
                best_metric = test_c_index
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
            
            print(
                f"best cindex: {best_metric:4f} "
                f"at epoch: {best_metric_epoch}"
            )

            # early_stopping needs the validation loss to check if it has decresed
            early_stopping(test_c_index, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break                    
        
        save_lr_history.append(optimizer.param_groups[0]['lr'])
        np.save(os.path.join(model_dir, 'lr_history.npy'), save_lr_history)
        scheduler.step(test_epoch_loss)

    

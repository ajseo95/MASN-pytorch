# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------

import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import h5py

seed = 999

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def _init_fn(worker_id):
    np.random.seed(seed)


from data_utils.dataset import TGIFQA
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from model.masn import MASN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Main script."""

    args.pin_memory = False
    args.dataset = 'tgif_qa'
    args.log = './logs/%s' % args.model_name
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    args.val_epoch_step = 1

    args.save_model_path = os.path.join(args.save_path, args.model_name)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    full_dataset = TGIFQA(
        dataset_name='train',
        q_max_length=args.q_max_length,
        v_max_length=args.v_max_length,
        max_n_videos=args.max_n_videos,
        data_type=args.task,
        csv_dir=args.df_dir,
        vocab_dir=args.vc_dir,
        feat_dir=args.feat_dir)
    test_dataset = TGIFQA(
        dataset_name='test',
        q_max_length=args.q_max_length,
        v_max_length=args.v_max_length,
        max_n_videos=args.max_n_videos,
        data_type=args.task,
        csv_dir=args.df_dir,
        vocab_dir=args.vc_dir,
        feat_dir=args.feat_dir)

    val_size = int(args.val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    print(
        'Dataset lengths train/val/test %d/%d/%d' %
        (len(train_dataset), len(val_dataset), len(test_dataset)))

    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=_init_fn)
    val_dataloader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=_init_fn)
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=_init_fn)

    print('Load data successful.')

    args.resnet_input_size = 2048
    args.i3d_input_size = 2048

    args.text_embed_size = train_dataset.dataset.GLOVE_EMBEDDING_SIZE
    args.answer_vocab_size = None

    args.word_matrix = train_dataset.dataset.word_matrix
    args.voc_len = args.word_matrix.shape[0]
    assert args.text_embed_size == args.word_matrix.shape[1]

    VOCABULARY_SIZE = train_dataset.dataset.n_words
    assert VOCABULARY_SIZE == args.voc_len

    ### criterions
    if args.task == 'Count':
        # add L2 loss
        criterion = nn.MSELoss().to(device)
    elif args.task in ['Action', 'Trans']:
        from embed_loss import MultipleChoiceLoss
        criterion = MultipleChoiceLoss(
            num_option=5, margin=1, size_average=True).to(device)
    elif args.task == 'FrameQA':
        # add classification loss
        args.answer_vocab_size = len(train_dataset.dataset.ans2idx)
        print(('Vocabulary size', args.answer_vocab_size, VOCABULARY_SIZE))
        criterion = nn.CrossEntropyLoss().to(device)

    if not args.test:
        train(
            args, train_dataloader, val_dataloader, test_dataloader, criterion)
    else:
        print(args.checkpoint[:5], args.task[:5])
        model = torch.load(os.path.join(args.save_model_path, args.checkpoint))
        test(args, model, test_dataloader, 0, criterion)


def train(args, train_dataloader, val_dataloader, test_dataloader, criterion):
    model = MASN(
        args.voc_len,
        args.rnn_layers,
        args.word_matrix,
        args.resnet_input_size,
        args.i3d_input_size,
        args.hidden_size,
        dropout_p=args.dropout,
        gcn_layers=args.gcn_layers,
        answer_vocab_size=args.answer_vocab_size,
        q_max_len=args.q_max_length,
        v_max_len=args.v_max_length,
        ablation=args.ablation)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if args.change_lr == 'none':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.change_lr == 'acc':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'loss':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'cos':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # consine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'step':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_list, gamma=0.1)
        # scheduler_warmup = GradualWarmupScheduler(
        #     optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)

    best_val_acc = 0. if args.task != 'Count' else -100.

    for epoch in range(args.max_epoch):
        print('Start Training Epoch: {}'.format(epoch))

        model.train()

        loss_list = []
        prediction_list = []
        correct_answer_list = []

        if args.change_lr == 'cos':
            # consine annealing
            scheduler_warmup.step(epoch=epoch)

        for ii, data in enumerate(train_dataloader):
            if epoch == 0 and ii == 0:
                print([d.dtype for d in data], [d.size() for d in data])
            data = [d.to(device) for d in data]

            optimizer.zero_grad()
            out, predictions, answers = model(args.task, *data)
            loss = criterion(out, answers)
            loss.backward()
            optimizer.step()

            correct_answer_list.append(answers)
            loss_list.append(loss.item())
            prediction_list.append(predictions.detach())
            if ii % 100 == 0:
                print("Batch: ", ii)

        train_loss = np.mean(loss_list)

        correct_answer = torch.cat(correct_answer_list, dim=0).long()
        predict_answer = torch.cat(prediction_list, dim=0).long()
        assert correct_answer.shape == predict_answer.shape

        current_num = torch.sum(predict_answer == correct_answer).cpu().numpy()
        acc = current_num / len(correct_answer) * 100.

        # print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        if args.change_lr == 'acc':
            scheduler_warmup.step(epoch, val_acc)
        elif args.change_lr == 'loss':
            scheduler_warmup.step(epoch, val_loss)
        elif args.change_lr == 'step':
            scheduler.step()

        print(
            "Train|Epoch: {}, Acc : {:.3f}={}/{}, Train Loss: {:.3f}".format(
                epoch, acc, current_num, len(correct_answer), train_loss))
        if args.task == 'Count':
            count_loss = F.mse_loss(
                predict_answer.float(), correct_answer.float())
            print('Train|Count Real Loss:\t {:.3f}'.format(count_loss))

        logfile = open(os.path.join(args.log, args.task + '.txt'), 'a+')
        logfile.write(
            "Train|Epoch: %d, Acc : %.3f=%d/%d, Train Loss: %.3f\n"
            % (epoch, acc, current_num, len(correct_answer), train_loss)
        )
        if args.task == 'Count':
            logfile.write(
                "Train|Count Real Loss:\t %.3f\n"%count_loss
            )
        logfile.close()

        val_acc, val_loss = val(args, model, val_dataloader, epoch, criterion)

        if val_acc > best_val_acc:
            print('Best Val Acc ======')
            best_val_acc = val_acc
        if epoch % args.val_epoch_step == 0 or val_acc >= best_val_acc:
            test(args, model, test_dataloader, epoch, criterion)


@torch.no_grad()
def val(args, model, val_dataloader, epoch, criterion):
    model.eval()

    loss_list = []
    prediction_list = []
    correct_answer_list = []

    for ii, data in enumerate(val_dataloader):
        data = [d.to(device) for d in data]

        out, predictions, answers = model(args.task, *data)
        loss = criterion(out, answers)

        correct_answer_list.append(answers)
        loss_list.append(loss.item())
        prediction_list.append(predictions.detach())

    val_loss = np.mean(loss_list)
    correct_answer = torch.cat(correct_answer_list, dim=0).long()
    predict_answer = torch.cat(prediction_list, dim=0).long()
    assert correct_answer.shape == predict_answer.shape

    current_num = torch.sum(predict_answer == correct_answer).cpu().numpy()

    acc = current_num / len(correct_answer) * 100.

    print(
        "VAL|Epoch: {}, Acc: {:3f}={}/{}, Val Loss: {:3f}".format(
            epoch, acc, current_num, len(correct_answer), val_loss))

    logfile = open(os.path.join(args.log, args.task + '.txt'), 'a+')
    logfile.write(
        "VAL|Epoch: %d, Acc: %.3f=%d/%d, Val Loss: %.3f\n"
        % (epoch, acc, current_num, len(correct_answer), val_loss)
    )
    logfile.close()

    if args.task == 'Count':
        print(
            'VAL|Count Real Loss:\t {:.3f}'.format(
                F.mse_loss(predict_answer.float(), correct_answer.float())))
        acc = -F.mse_loss(predict_answer.float(), correct_answer.float())

        logfile = open(os.path.join(args.log, args.task + '.txt'), 'a+')
        logfile.write(
            "VAL|Count Real Loss:\t %.3f}\n"
            % (F.mse_loss(predict_answer.float(), correct_answer.float()))
        )
        logfile.close()
    return acc, val_loss


@torch.no_grad()
def test(args, model, test_dataloader, epoch, criterion):

    model.eval()

    loss_list = []
    prediction_list = []
    correct_answer_list = []

    for ii, data in enumerate(test_dataloader):
        data = [d.to(device) for d in data]

        out, predictions, answers = model(args.task, *data)
        loss = criterion(out, answers)

        correct_answer_list.append(answers)
        loss_list.append(loss.item())
        prediction_list.append(predictions.detach())

    test_loss = np.mean(loss_list)
    correct_answer = torch.cat(correct_answer_list, dim=0).long()
    predict_answer = torch.cat(prediction_list, dim=0).long()
    assert correct_answer.shape == predict_answer.shape

    current_num = torch.sum(predict_answer == correct_answer).cpu().numpy()

    acc = current_num / len(correct_answer) * 100.

    print(
        "Test|Epoch: {}, Acc: {:3f}={}/{}, Test Loss: {:3f}".format(
            epoch, acc, current_num, len(correct_answer), test_loss))

    logfile = open(os.path.join(args.log, args.task + '.txt'), 'a+')
    logfile.write(
        "Test|Epoch: %d, Acc: %.3f=%d/%d, Test Loss: %.3f\n"
        % (epoch, acc, current_num, len(correct_answer), test_loss)
    )
    logfile.close()

    if args.save:
        if (args.task == 'Action' and
            acc >= 80) or (args.task == 'Trans' and
                           acc >= 80) or (args.task == 'FrameQA' and
                                          acc >= 55):
            torch.save(
                model, os.path.join(args.save_model_path,
                                    args.task + '_' + str(acc.item())[:5] + '.pth'))
            print('Save model at ', args.save_model_path)

    if args.task == 'Count':
        count_loss = F.mse_loss(predict_answer.float(), correct_answer.float())
        print('Test|Count Real Loss:\t {:.3f}'.format(count_loss))
        logfile = open(os.path.join(args.log, args.task + '.txt'), 'a+')
        logfile.write(
            'Test|Count Real Loss:\t %.3f\n' % (count_loss)
        )
        logfile.close()
        if args.save and count_loss <= 4.0:
            torch.save(
                model, os.path.join(args.save_model_path,
                                    args.task + '_' + str(count_loss.item())[:5] + '.pth'))
            print('Save model at ', args.save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='[MASN]')
    
    ################ path config ################
    parser.add_argument('--feat_dir', default='/data/TGIFQA', help='path for resnet and i3d features')
    parser.add_argument('--vc_dir', default='/data/TGIFQA/vocab', help='path for vocabulary')
    parser.add_argument('--df_dir', default='/data/TGIFQA/question', help='path for tgif question csv files')

    ################ inference config ################
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='Action_84.38.pth',
        help='path to checkpoint')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./saved_models/',
        help='path for saving trained models')

    parser.add_argument(
        '--save', action='store_true', default=True, help='save models or not')
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='dimension of model')
    parser.add_argument(
        '--test', action='store_true', default=False, help='Train or Test')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--q_max_length', type=int, default=20)
    parser.add_argument('--v_max_length', type=int, default=20)

    parser.add_argument(
        '--task',
        type=str,
        default='Count',
        help='[Count, Action, FrameQA, Trans]')
    parser.add_argument(
        '--rnn_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument(
        '--gcn_layers',
        type=int,
        default=2,
        help='number of layers in gcn (+1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_n_videos', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_list', type=list, default=[10, 20, 30, 40])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument(
        '--change_lr', type=str, default='none', help='0 False, 1 True')
    parser.add_argument(
        '--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--ablation', type=str, default='none')

    args = parser.parse_args()
    print(args)

    main()
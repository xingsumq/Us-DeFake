from utility.globals import *
from embedding.text_model import GraphSAINT
from embedding.user_model import UnGraphSAINT
from embedding.minibatch import Minibatch
from utility.utils import *
from utility.metric import *
import torch
import time
import torch.nn.functional as F


def evaluate_full_batch(model, minibatch, loss, preds, labels, mode=''):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).
    """
    if mode == 'val':
        node_target = [minibatch.node_val]
    elif mode == 'test':
        node_target = [minibatch.node_test]
    else:
        assert mode == 'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]
    acc, pre, rec, f1 = [], [], [], []
    for n in node_target:
        results = evaluation(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        acc.append(results[0])
        pre.append(results[1])
        rec.append(results[2])
        f1.append(results[3])
    acc = acc[0] if len(acc) == 1 else acc
    pre = pre[0] if len(pre) == 1 else pre
    rec = rec[0] if len(rec) == 1 else rec
    f1 = f1[0] if len(f1) == 1 else f1
    # loss is not very accurate in this case, since loss is also contributed by training nodes
    # on the other hand, for val / test, we mostly care about their accuracy only.
    # so the loss issue is not a problem.
    return loss, acc, pre, rec, f1


def evaluate_source_news(model, minibatch, preds_vate, labels_vate, mode=''):
    role = get_source_news(args_global.data_prefix,args_global.fold)
    assert mode == 'valtest'
    node_target = [minibatch.node_val, minibatch.node_test]
    source_target = []
    for nodes in node_target:
        souce = []
        for n in nodes:
            if n in role:
                souce.append(n)
        source_target.append(souce)
    print(source_target)

    acc, pre, rec, f1 = [], [], [], []
    for n in source_target:
        results = evaluation(to_numpy(labels_vate[n]), to_numpy(preds_vate[n]), model.sigmoid_loss)
        acc.append(results[0])
        pre.append(results[1])
        rec.append(results[2])
        f1.append(results[3])
    acc = acc[0] if len(acc) == 1 else acc
    pre = pre[0] if len(pre) == 1 else pre
    rec = rec[0] if len(rec) == 1 else rec
    f1 = f1[0] if len(f1) == 1 else f1

    acc_val, acc_test = acc
    pre_val, pre_test = pre
    rec_val, rec_test = rec
    f1_val, f1_test = f1
    return acc_val, acc_test, pre_val, pre_test, rec_val, rec_test, f1_val, f1_test


def text_prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS in text model = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def user_prepare(train_data,train_params,arch_gcn):
    adj_full, adj_train, feat_full, role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = UnGraphSAINT(arch_gcn, train_params, feat_full)
    printf("TOTAL NUM OF PARAMS in user model = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=UnGraphSAINT(arch_gcn, train_params, feat_full, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train_step(model_t, minibatch_t, model_u, minibatch_u):
    node_subgraph_t, adj_subgraph_t, norm_loss_subgraph_t = minibatch_t.one_batch(mode='train')
    node_subgraph_u, adj_subgraph_u, norm_loss_subgraph_u = minibatch_u.one_batch(mode='train')

    model_t.train()
    model_u.train()
    model_t.optimizer.zero_grad()
    model_u.optimizer.zero_grad()

    t_preds_train, labels_train, labels_converted = model_t.forward(node_subgraph_t, adj_subgraph_t)
    loss_t = model_t._loss(t_preds_train, labels_converted, norm_loss_subgraph_t)

    u_preds_train = model_u.forward(node_subgraph_u, adj_subgraph_u)
    loss_u = model_u.rec_loss(adj_subgraph_u, u_preds_train, norm_loss_subgraph_u)
    if args_global.gpu >= 0:
        loss_u = loss_u.cuda()
    loss_train = loss_u + loss_t

    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model_t.parameters(), 5)
    torch.nn.utils.clip_grad_norm_(model_u.parameters(), 5)
    model_t.optimizer.step()
    model_u.optimizer.step()
    pred_train = emb_concatenation(t_preds_train, u_preds_train, node_subgraph_u, args_global)
    preds_train = F.softmax(pred_train, dim=1)

    return loss_train, preds_train, labels_train


def eval_step(model_eval_t, minibatch_t, model_eval_u, minibatch_u, mode=''):
    node_subgraph_eval_t, adj_subgraph_eval_t, norm_loss_subgraph_eval_t = minibatch_t.one_batch(mode=mode)
    node_subgraph_eval_u, adj_subgraph_eval_u, norm_loss_subgraph_eval_u = minibatch_u.one_batch(mode=mode)

    model_eval_t.eval()
    model_eval_u.eval()
    with torch.no_grad():
        t_preds_eval, labels_eval, labels_converted_eval = model_eval_t.forward(node_subgraph_eval_t, adj_subgraph_eval_t)
        loss_t_eval = model_eval_t._loss(t_preds_eval, labels_converted_eval, norm_loss_subgraph_eval_t)

        u_preds_eval = model_eval_u.forward(node_subgraph_eval_u, adj_subgraph_eval_u)
        loss_u_eval = model_eval_u.rec_loss(adj_subgraph_eval_u, u_preds_eval, norm_loss_subgraph_eval_u)
        loss_eval = loss_t_eval + loss_u_eval
        pred_eval = emb_concatenation(t_preds_eval, u_preds_eval, node_subgraph_eval_u, args_global)
        preds_eval = F.softmax(pred_eval, dim=1)
    return loss_eval, preds_eval, labels_eval


def train(train_phases, t_model, t_minibatch, t_minibatch_eval, t_model_eval, u_model, u_minibatch, u_minibatch_eval, u_model_eval, eval_val_every):
    if not args_global.cpu_eval:
        t_minibatch_eval=t_minibatch
        u_minibatch_eval=u_minibatch
    epoch_ph_start = 0
    acc_best, ep_best = 0, -1
    time_train = 0
    dir_saver = '{}/saved_models'.format(args_global.dir_log)
    path_saver_t = '{}/saved_models/saved_model_text_{}.pkl'.format(args_global.dir_log, timestamp)
    path_saver_u = '{}/saved_models/saved_model_user_{}.pkl'.format(args_global.dir_log, timestamp)

    # for ip, phase in enumerate(train_phases):
    #     printf('START PHASE {:4d}'.format(ip),style='underline')
    phase = train_phases[0]
    t_minibatch.set_sampler(phase)
    u_minibatch.set_sampler(phase)
    t_num_batches = t_minibatch.num_training_batches()
    u_num_batches = u_minibatch.num_training_batches()
    for e in range(epoch_ph_start, int(phase['end'])):
        printf('Epoch {:4d}'.format(e),style='bold')
        t_minibatch.shuffle()
        u_minibatch.shuffle()
        l_loss_tr, l_acc_tr, l_pre_tr, l_rec_tr, l_f1_tr  = [], [], [], [], []
        time_train_ep = 0
        while not t_minibatch.end():
            t1 = time.time()
            loss_train,preds_train,labels_train = train_step(t_model,t_minibatch,u_model,u_minibatch)
            time_train_ep += time.time() - t1
            if not t_minibatch.batch_num % args_global.eval_train_every:
                acc, pre, rec, f1 = evaluation(to_numpy(labels_train),to_numpy(preds_train),t_model.sigmoid_loss)
                l_loss_tr.append(loss_train)
                l_acc_tr.append(acc)
                l_pre_tr.append(pre)
                l_rec_tr.append(rec)
                l_f1_tr.append(f1)
        if (e+1)%eval_val_every == 0:
            if args_global.cpu_eval:
                torch.save(t_model.state_dict(),'t_tmp.pkl')
                torch.save(u_model.state_dict(), 'u_tmp.pkl')
                t_model_eval.load_state_dict(torch.load('t_tmp.pkl',map_location=lambda storage, loc: storage))
                u_model_eval.load_state_dict(torch.load('u_tmp.pkl', map_location=lambda storage, loc: storage))
            else:
                t_model_eval = t_model
                u_model_eval = u_model

            loss_eval, preds_eval, labels_eval = eval_step(t_model_eval, t_minibatch_eval, u_model_eval, u_minibatch_eval, mode='val')
            loss_val, acc_val, pre_val, rec_val, f1_val = evaluate_full_batch(t_model_eval, t_minibatch_eval, loss_eval,
                                                                              preds_eval, labels_eval, mode='val')
            printf('TRAIN (Ep avg): loss = {:.4f}\taccuracy = {:.4f}\tprecision = {:.4f}\trecall = {:.4f}\tF1  = {:.4f}\ttrain time = {:.4f} sec'\
                    .format(f_mean(l_loss_tr), f_mean(l_acc_tr), f_mean(l_pre_tr), f_mean(l_rec_tr), f_mean(l_f1_tr), time_train_ep))
            printf('VALIDATION: loss = {:.4f}\taccuracy = {:.4f}\tprecision = {:.4f}\trecall = {:.4f}\tF1  = {:.4f}'\
                    .format(loss_val, acc_val, pre_val, rec_val, f1_val), style='yellow')
            if acc_val > acc_best:
                acc_best, ep_best = acc_val, e
                if not os.path.exists(dir_saver):
                    os.makedirs(dir_saver)
                printf('  Saving model ...', style='yellow')
                torch.save(t_model.state_dict(), path_saver_t)
                torch.save(u_model.state_dict(), path_saver_u)
        time_train += time_train_ep
    epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if args_global.cpu_eval:
            t_model_eval.load_state_dict(torch.load(path_saver_t, map_location=lambda storage, loc: storage))
            u_model_eval.load_state_dict(torch.load(path_saver_u, map_location=lambda storage, loc: storage))
        else:
            t_model.load_state_dict(torch.load(path_saver_t))
            u_model.load_state_dict(torch.load(path_saver_u))
            t_model_eval=t_model
            u_model_eval=u_model
        printf('  Restoring model ...', style='yellow')

    loss_vate, preds_vate, labels_vate = eval_step(t_model_eval,t_minibatch_eval,u_model_eval,u_minibatch_eval,mode='valtest')
    acc_val, acc_test, pre_val, pre_test, rec_val, rec_test, f1_val, f1_test = evaluate_source_news(t_model_eval, t_minibatch_eval,
                                                                              preds_vate, labels_vate, mode='valtest')

    printf("Full validation (Epoch {:4d}): \n Accuracy = {:.4f}\tPrecision = {:.4f}\tRecall = {:.4f}\tF1 = {:.4f}"\
            .format(ep_best, acc_val, pre_val, rec_val, f1_val), style='red')
    printf("Full test stats: \n  Accuracy = {:.4f}\tPrecision = {:.4f}\tRecall = {:.4f}\tF1 = {:.4f}"\
            .format(acc_test, pre_test, rec_test, f1_test), style='red')


if __name__ == '__main__':
    log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)
    train_params, train_phases, arch_gcn = parse_n_prepare(args_global)
    text_train_data, user_train_data = data_prepare(args_global)
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP

    t_model, t_minibatch, t_minibatch_eval, t_model_eval = text_prepare(text_train_data, train_params, arch_gcn)
    u_model, u_minibatch, u_minibatch_eval, u_model_eval = user_prepare(user_train_data, train_params, arch_gcn)
    train(train_phases, t_model, t_minibatch, t_minibatch_eval, t_model_eval, u_model, u_minibatch, u_minibatch_eval, u_model_eval, train_params['eval_val_every'])

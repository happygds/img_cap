from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts_mix as opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_cider_scorer, get_self_critical_reward, c2f_get_self_critical_reward

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories.pkl')):
            with open(os.path.join(opt.start_from, 'histories.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    init_iteration = iteration

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    if opt.gamma < 0:
        gamma = 0
    else:
        gamma = opt.gamma

    if opt.caption_model == 'c2ftopdown' or opt.caption_model == 'c2fada':
        crit = utils.c2fLanguageModelCriterion(gamma)
    else:
        crit = utils.LanguageModelCriterion(gamma)
    rl_crit = utils.RewardCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        # # eval model
        # eval_kwargs = {'split': 'val', 'verbose': True,
        #                'dataset': opt.input_json}
        # eval_kwargs.update(vars(opt))
        # _, _, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)
        # print('before train: ', lang_stats)

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_cider_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
            alpha = 0.9 ** max(0., (epoch - 33.))
            # alpha = 1.

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        if iteration % 5 == 0:
            print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        else:
            if opt.caption_model == 'c2ftopdown' or opt.caption_model == 'c2fada':
                gen_result, sample_logprobs, gen_result_fine, sample_logprobs_fine = model.sample(
                    fc_feats, att_feats, {'sample_max': 0, 'temperature': opt.temperature})

                reward, reward_fine = c2f_get_self_critical_reward(
                    model, fc_feats, att_feats, data, gen_result, gen_result_fine, alpha, only_cider=opt.only_cider)
                loss = rl_crit(sample_logprobs[0], gen_result, Variable(
                    torch.from_numpy(reward).float().cuda(), requires_grad=False))
                loss += rl_crit(sample_logprobs_fine[0], gen_result_fine, Variable(
                    torch.from_numpy(reward_fine).float().cuda(), requires_grad=False))

                # loss += 5e-3 * crit([sample_logprobs_fine[1], sample_logprobs[1]], labels[:, 1:], masks[:, 1:])
            else:
                gen_result, sample_logprobs = model.sample(
                    fc_feats, att_feats, {'sample_max': 0, 'temperature': opt.temperature})
                reward = get_self_critical_reward(model, fc_feats, att_feats, data, gen_result, only_cider=opt.only_cider)
                loss = rl_crit(sample_logprobs, gen_result, Variable(
                    torch.from_numpy(reward).float().cuda(), requires_grad=False))

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        if iteration % 5 == 0:
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), train_loss = {:.3f}, avg_reward = {:.3f}, time/batch = {:.3f}"
                      .format(iteration, epoch, train_loss, np.mean(reward[:, 0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tf_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # Save model if is improving on validation result
        if opt.id.isdigit():
            model_id = (iteration - init_iteration) // opt.save_checkpoint_every + int(opt.id) + 1
        else:
            model_id = (iteration - init_iteration) // opt.save_checkpoint_every

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val', 'verbose': True,
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs, model_id=model_id)
            if opt.caption_model == 'c2ftopdown' or opt.caption_model == 'c2fada':
                lang_stats, lang_stats_fine = lang_stats
            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            with open(os.path.join(opt.checkpoint_path, 'val.RandB.scores.txt'), 'a+') as f:
                f.write('\n')
                f.write("Model {}:".format(model_id))
                f.write('\n')
                f.write("Bleu_1:" + str(lang_stats['Bleu_1']))
                f.write('\n')
                f.write("Bleu_2:" + str(lang_stats['Bleu_2']))
                f.write('\n')
                f.write("Bleu_3:" + str(lang_stats['Bleu_3']))
                f.write('\n')
                f.write("Bleu_4:" + str(lang_stats['Bleu_4']))
                f.write('\n')
                # f.write("SPICE:" + str(scores['SPICE']))
                # f.write('\n')
                f.write("ROUGE_L:" + str(lang_stats['ROUGE_L']))
                f.write('\n')
                f.write("Meteor:" + str(lang_stats['METEOR']))
                f.write('\n')
                f.write("CIDEr:" + str(lang_stats['CIDEr']))
                f.write('\n')
                f.write("metric " + str(2. * lang_stats['Bleu_4'] + lang_stats['CIDEr'] +
                                        5. * lang_stats['ROUGE_L'] + 10. * lang_stats['METEOR']))
                f.write('\n\n')

            if opt.caption_model == 'c2ftopdown' or opt.caption_model == 'c2fada':
                with open(os.path.join(opt.checkpoint_path, 'val.RandB.scores_fine.txt'), 'a+') as f:
                    f.write('\n')
                    f.write("Model {}:".format(model_id))
                    f.write('\n')
                    f.write("Bleu_1:" + str(lang_stats_fine['Bleu_1']))
                    f.write('\n')
                    f.write("Bleu_2:" + str(lang_stats_fine['Bleu_2']))
                    f.write('\n')
                    f.write("Bleu_3:" + str(lang_stats_fine['Bleu_3']))
                    f.write('\n')
                    f.write("Bleu_4:" + str(lang_stats_fine['Bleu_4']))
                    f.write('\n')
                    # f.write("SPICE:" + str(scores['SPICE']))
                    # f.write('\n')
                    f.write("ROUGE_L:" + str(lang_stats_fine['ROUGE_L']))
                    f.write('\n')
                    f.write("Meteor:" + str(lang_stats_fine['METEOR']))
                    f.write('\n')
                    f.write("CIDEr:" + str(lang_stats_fine['CIDEr']))
                    f.write('\n')
                    f.write("metric " + str(2. * lang_stats_fine['Bleu_4'] + lang_stats_fine['CIDEr'] +
                                            5. * lang_stats_fine['ROUGE_L'] + 10. * lang_stats_fine['METEOR']))
                    f.write('\n\n')

            if True:  # if true

                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'histories.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-{}.pth'.format(model_id))
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_{}.pkl'.format(model_id)), 'wb') as f:
                    cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
train(opt)

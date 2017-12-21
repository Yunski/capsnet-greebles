import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from config import cfg
from utils import load_data

from capsnet import CapsNet


def saver(dataset, is_training=True):
    if not os.path.exists(cfg.summary_dir):
        os.mkdir(cfg.summary_dir)
    summary_dir = os.path.join(cfg.summary_dir, dataset)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    if is_training:
        loss = summary_dir + '/loss.csv'
        train_err = summary_dir + '/train_err.csv'
        val_err = summary_dir + '/val_err.csv'

        if os.path.exists(val_err):
            os.remove(val_err)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_err):
            os.remove(train_err)

        fd_train_err = open(train_err, 'w')
        fd_train_err.write('step,train_err\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_err = open(val_err, 'w')
        fd_val_err.write('step,val_err\n')
        return fd_train_err, fd_loss, fd_val_err
    else:
        test_err = summary_dir + '/test_err.csv'
        if os.path.exists(test_err):
            os.remove(test_err)
        fd_test_err = open(test_err, 'w')
        fd_test_err.write('test_err\n')
        return fd_test_err

def train(model, supervisor, dataset):
    data = load_data(dataset, cfg.batch_size, samples_per_epoch=cfg.samples_per_epoch)
    if not data:
        raise ValueError("{} is not an available dataset".format(dataset))
    X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches = data
    fd_train_err, fd_loss, fd_val_err = saver(dataset)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print("hi")
    with supervisor.managed_session(config=config) as sess:
        for epoch in range(cfg.epochs):
            sys.stdout.write("Epoch {}/{}\n".format(epoch + 1, cfg.epochs)) 
            sys.stdout.flush()
            if supervisor.should_stop():
                print('supervisor stopped!')
                break
            progress_bar = tqdm(range(num_train_batches), total=num_train_batches, ncols=70, leave=False, unit='b')
            epoch_train_err = 0
            epoch_val_err = 0
            for step in progress_bar:    
                global_step = epoch * num_train_batches + step
                train_err = 0
                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_err, summary_str = sess.run([model.train_op, 
                                                                model.total_loss, 
                                                                model.error_rate, 
                                                                model.train_summary])
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_err.write(str(global_step) + ',' + str(train_err) + "\n")
                    fd_train_err.flush()
                else:
                    _, train_err = sess.run([model.train_op, model.error_rate])

                epoch_train_err = (epoch_train_err * step + train_err) / (step + 1)
                progress_bar.set_description("\rtrain_err: {:.4f} - train_acc: {:.4f}".format(epoch_train_err, 1 - epoch_train_err))

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_err = 0
                    for i in range(num_val_batches):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        err = sess.run(model.error_rate, {model.X: X_val[start:end], model.labels: Y_val[start:end]})
                        val_err += err
                    val_err = val_err / num_val_batches
                    fd_val_err.write(str(global_step) + ',' + str(val_err) + '\n')
                    fd_val_err.flush()

            progress_bar = tqdm(range(num_val_batches), total=num_val_batches, ncols=70, leave=False, unit='b')
            for step in progress_bar:
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                err = sess.run(model.error_rate, {model.X: X_val[start:end], model.labels: Y_val[start:end]})
                epoch_val_err = (epoch_val_err * step + err) / (step + 1)
                progress_bar.set_description("\rval_err: {:.4f} - val_acc: {:.4f}".format(epoch_val_err, 1 - epoch_val_err))
    
            sys.stdout.write("train_err: {:.4f} - train_acc: {:.4f} - val_err: {:.4f} - val_acc: {:.4f}\n".format(epoch_train_err, 1 - epoch_train_err, epoch_val_err, 1 - epoch_val_err))
            sys.stdout.flush() 
            
            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + "/{}/model_epoch_{:04d}_step_{:02d}".format(dataset, epoch, global_step))

        fd_val_err.close()
        fd_train_err.close()
        fd_loss.close()


def evaluate(model, supervisor, dataset):
    data = load_data(dataset, cfg.batch_size, is_training=False)
    if not data:
        raise ValueError("{} is not an available dataset".format(dataset))
    X_test, Y_test, num_test_batches = data
    fd_test_err = saver(dataset, is_training=False)
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(os.path.join(cfg.logdir, dataset)))
        tf.logging.info('Model restored!')
        test_err = 0
        progress_bar = tqdm(range(num_test_batches), total=num_test_batches, ncols=70, leave=False, unit='b')
        for step in progress_bar:    
            start = step * cfg.batch_size
            end = start + cfg.batch_size
            err = sess.run(model.error_rate, {model.X: X_test[start:end], model.labels: Y_test[start:end]})
            test_err = (test_err * step + err) / (step + 1)
            progress_bar.set_description("\r>> test_err: {:.4f} - test_acc: {:.4f}".format(test_err, 1 - test_err))
        sys.stdout.write("Final - test_err: {:.4f} - test_acc: {:.4f}\n".format(test_err, 1 - test_err))
        sys.stdout.flush()
        fd_test_err.write(str(test_err) + "\n")
        fd_test_err.close()


def main(_):
    dataset = cfg.dataset
    if dataset == 'mnist':
        input_shape = (cfg.batch_size, 28, 28, 1)
    elif dataset == 'affnist':
        input_shape = (cfg.batch_size, 40, 40, 1)
    else:
        raise ValueError("{} is not an available dataset".format(dataset))

    tf.logging.info("Initializing CapsNet for {}...".format(dataset))
    model = CapsNet(input_shape, is_training=cfg.is_training)
    tf.logging.info("Finished initialization.")

    logdir = os.path.join(cfg.logdir, dataset)
    sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)
    if cfg.is_training:
        tf.logging.info("Initialize training...")
        train(model, sv, dataset)
        tf.logging.info("Finished training.")
    else:
        tf.logging.info("Initialize evaluation...")
        evaluate(model, sv, dataset)
        tf.logging.info("Finished evaluation.")

if __name__ == '__main__':
    tf.app.run()

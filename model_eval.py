import os
import sys
import tensorflow as tf

from tqdm import tqdm

from utils import load_data
from config import cfg

"""
Adapted from naturomics/CapsNet-Tensorflow
"""

def train(model, supervisor, dataset):
    data = load_data(dataset, cfg.batch_size, samples_per_epoch=cfg.samples_per_epoch, use_val_only=True)
    if not data:
        raise ValueError("{} is not an available dataset".format(dataset))
    X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches = data
    fd_train_err, fd_loss, fd_val_err = saver(cfg.summary_dir, model.name, dataset)
    logdir = os.path.join(os.path.join(cfg.logdir, model.name), dataset)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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
            epoch_loss = 0
            for step in progress_bar:
                global_step = epoch * num_train_batches + step
                train_err = 0
                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_err, summary_str = sess.run([model.train_op,
                                                                model.total_loss,
                                                                model.error_rate,
                                                                model.train_summary])
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write("{},{:.4f}\n".format(global_step, epoch_loss))
                    fd_loss.flush()
                    fd_train_err.write("{},{:.4f}\n".format(global_step, epoch_train_err))
                    fd_train_err.flush()
                else:
                    _, loss, train_err = sess.run([model.train_op, model.total_loss, model.error_rate])

                epoch_loss = (epoch_loss * step + loss) / (step + 1)
                epoch_train_err = (epoch_train_err * step + train_err) / (step + 1)
                progress_bar.set_description("\rtrain_err: {:.4f} - train_acc: {:.4f}".format(epoch_train_err, 1 - epoch_train_err))

            progress_bar = tqdm(range(num_val_batches), total=num_val_batches, ncols=70, leave=False, unit='b')
            for step in progress_bar:
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                err = sess.run(model.error_rate, {model.X: X_val[start:end], model.labels: Y_val[start:end]})
                epoch_val_err = (epoch_val_err * step + err) / (step + 1)
                progress_bar.set_description("\rval_err: {:.4f} - val_acc: {:.4f}".format(epoch_val_err, 1 - epoch_val_err))

            fd_val_err.write("{},{:.4f}\n".format(epoch, epoch_val_err))
            fd_val_err.flush()

            sys.stdout.write("train_err: {:.4f} - train_acc: {:.4f}".format(epoch_train_err, 1 - epoch_train_err))
            if num_val_batches > 0:
                sys.stdout.write(" - val_err: {:.4f} - val_acc: {:.4f}\n".format(epoch_val_err, 1 - epoch_val_err))
            else:
                sys.stdout.write("\n")
            sys.stdout.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, logdir + "/model_epoch_{:04d}_step_{:02d}".format(epoch, global_step))

        fd_val_err.close()
        fd_train_err.close()
        fd_loss.close()


def evaluate(model, supervisor, dataset):
    data = load_data(dataset, cfg.test_batch_size, is_training=False)
    if not data:
        raise ValueError("{} is not an available dataset".format(dataset))
    X_test, Y_test, num_test_batches = data
    fd_test_err = saver(cfg.summary_dir, model.name, dataset, is_training=False)
    logdir = os.path.join(os.path.join(cfg.logdir, model.name), dataset)
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(logdir))
        tf.logging.info('Model restored!')
        test_err = 0
        progress_bar = tqdm(range(num_test_batches), total=num_test_batches, ncols=70, leave=False, unit='b')
        for step in progress_bar:
            if supervisor.should_stop(): 
                break
            if len(X_test) == 0:
                if step % 100 == 0:
                    err, summary_str = sess.run([model.error_rate, model.train_summary])
                    supervisor.summary_writer.add_summary(summary_str, step)
                else:
                    err = sess.run(model.error_rate)
            else:
                start = step * cfg.test_batch_size
                end = start + cfg.test_batch_size
                err = sess.run(model.error_rate, {model.X: X_test[start:end], model.labels: Y_test[start:end]})
            test_err = (test_err * step + err) / (step + 1)
            progress_bar.set_description("\r>> test_err: {:.4f} - test_acc: {:.4f}".format(test_err, 1 - test_err))
        sys.stdout.write("Final - test_err: {:.4f} - test_acc: {:.4f}\n".format(test_err, 1 - test_err))
        sys.stdout.flush()
        fd_test_err.write("{:.4f}\n".format(test_err))
        fd_test_err.close()


def saver(summary_dir, model, dataset, is_training=True):
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    summary_dir = os.path.join(summary_dir, model)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    summary_dir = os.path.join(summary_dir, dataset)
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
        fd_val_err.write('epoch,val_err\n')
        return fd_train_err, fd_loss, fd_val_err
    else:
        test_err = summary_dir + '/test_err.csv'
        if os.path.exists(test_err):
            os.remove(test_err)
        fd_test_err = open(test_err, 'w')
        fd_test_err.write('test_err\n')
        return fd_test_err

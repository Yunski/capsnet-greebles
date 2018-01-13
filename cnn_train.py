import os
import tensorflow as tf

from config import cfg
from model_eval import train
from cnn_baseline import CNN
from utils import get_dataset_values

def main(_):
    dataset = cfg.dataset
    input_shape, num_classes = get_dataset_values(dataset, cfg.batch_size)
 
    tf.logging.info("Initializing CNN for {}...".format(dataset))
    model = CNN(input_shape, num_classes, is_training=True)
    tf.logging.info("Finished initialization.")

    if not os.path.exists(cfg.logdir):
        os.mkdir(cfg.logdir)
    logdir = os.path.join(cfg.logdir, model.name)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir = os.path.join(logdir, dataset)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=0)

    tf.logging.info("Initialize training...")
    train(model, sv, dataset)
    tf.logging.info("Finished training.")

    
if __name__ == '__main__':
    tf.app.run()
    
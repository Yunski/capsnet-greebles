import os
import tensorflow as tf

from config import cfg
from model_eval import evaluate 
from resnet import resnet
from utils import get_dataset_values

def main(_):
    dataset = cfg.dataset
    input_shape, num_classes, use_test_queue = get_dataset_values(dataset, cfg.test_batch_size, is_training=False)
 
    tf.logging.info("Initializing ResNet for {}...".format(dataset))
    model = resnet(input_shape, num_classes, is_training=False, use_test_queue=use_test_queue)
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

    tf.logging.info("Initialize evaluation...")
    evaluate(model, sv, dataset)
    tf.logging.info("Finished evaluation.")

    
if __name__ == '__main__':
    tf.app.run()
   

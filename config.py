import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 32, "batch size")
flags.DEFINE_integer('test_batch_size', 100, "batch size")
flags.DEFINE_integer('epochs', 80, 'epochs')
flags.DEFINE_integer('samples_per_epoch', 32000, 'samples per epoch')

flags.DEFINE_integer('num_threads', 8, "number of threads of enqueueing examples")
flags.DEFINE_integer('train_sum_freq', 100, "the frequency of saving train summary(step)")
flags.DEFINE_integer('val_sum_freq', 500, "the frequency of saving validation summary(step)")
flags.DEFINE_integer('save_freq', 5, "the frequency of saving model(epoch)")


flags.DEFINE_string('dataset', 'smallnorb', "the dataset")
flags.DEFINE_string('logdir', 'logs', "logs directory")
flags.DEFINE_string('summary_dir', 'summary', "path for saving model summary")
flags.DEFINE_string('dataset_file', 'datasets.yml', "file containing dataset urls")

cfg = tf.app.flags.FLAGS

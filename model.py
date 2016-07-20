from __future__ import division
from __future__ import print_function

import time
from datetime import timedelta
import argparse

import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import mi_rnn_cell
from tensorflow.python.ops import seq2seq
import numpy as np

from data_reader import DataReader


# Define or model
# ===============
class RNN_Model(object):
    def __init__(self, config):
        self.config = config

        # Main model structure
        if config.mi_type:
            cell = mi_rnn_cell.MIGRUCell(config.hidden_layer_size)
            dropped_cell = mi_rnn_cell.DropoutWrapper(
                cell, output_keep_prob=config.dropout)
            self.cell = cell = mi_rnn_cell.MultiRNNCell(
                [dropped_cell] * config.layers_qtty)
        else:
            cell = rnn_cell.GRUCell(config.hidden_layer_size)
            dropped_cell = rnn_cell.DropoutWrapper(
                cell, output_keep_prob=config.dropout)
            self.cell = cell = rnn_cell.MultiRNNCell(
                [dropped_cell] * config.layers_qtty)

        self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        self.input_x = tf.placeholder(
            tf.int32, [config.batch_size, config.sequence_size], "input_x")
        self.input_y = tf.placeholder(
            tf.int32, [config.batch_size, config.sequence_size], "input_y")

        with tf.variable_scope('rnnlm'):
            weight = tf.get_variable(
                "weight", [config.hidden_layer_size, config.vocab_size])
            bias = tf.get_variable("bias", [config.vocab_size])
            # split 2D tensor into list of tensors [[batch_size, 1]]
            # as input we send a list with len == sequence_size and
            # batch_size X 1 unit. really we just create rotated batches
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [config.vocab_size, config.hidden_layer_size])
                embeded = tf.nn.embedding_lookup(embedding, self.input_x)
                inputs = tf.split(1, config.sequence_size, embeded)
                inputs_list = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, self.final_state = rnn.rnn(
            self.cell, inputs_list, initial_state=self.initial_state)
        output = tf.reshape(tf.concat(1, outputs),
                            [-1, config.hidden_layer_size])
        self.logits = tf.matmul(output, weight) + bias
        self.prediction = tf.nn.softmax(self.logits)

        # loss/cost calculations
        self.loss = seq2seq.sequence_loss_by_example(
            logits=[self.logits],
            targets=[tf.reshape(self.input_y, [-1])],
            weights=[tf.ones([config.batch_size*config.sequence_size])])
        self.cost = tf.div(tf.div(
            tf.reduce_sum(self.loss), config.batch_size),
            config.sequence_size)

        # updatedoptimizer definition
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), config.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_operation = optimizer.apply_gradients(zip(grads, tvars))

        self.perplexity = tf.exp(self.cost / config.sequence_size)

        tf.histogram_summary("loss", self.loss)
        tf.scalar_summary("cost", self.cost)
        tf.scalar_summary("learning rate", self.learning_rate)
        tf.scalar_summary("perplexity", self.perplexity)
        self.merged = tf.merge_all_summaries()

    def get_sample(self, sess, chars, vocab, num_to_predict=200,
                   initial_sentence='The ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in initial_sentence[:-1]:
            x = np.array(vocab[char]).reshape(1, 1)
            feed = {self.input_x: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        predicted_text = initial_sentence
        char = initial_sentence[-1]
        for n in range(num_to_predict):
            x = np.array(vocab[char]).reshape(1, 1)
            feed = {self.input_x: x, self.initial_state: state}
            probs, state = sess.run([self.prediction, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            else:
                sample = weighted_pick(p)

            char = pred_letter = chars[sample]
            predicted_text += pred_letter
        return predicted_text


# Define config class
# ===================
class Config(object):
    learning_rate = 0.002
    # clip gradients at this value
    grad_clip = 5.0
    # decay rate for network
    decay_rate = 0.99
    # hidden layer number of features
    hidden_layer_size = 100
    # number of layers for multilayers RNN
    layers_qtty = 2
    # dropout for training
    dropout = 0.7
    # number of epochs
    num_epochs = 5000
    # batch size for reader
    batch_size = 50
    # letter chunk size at batch
    sequence_size = 50
    # how many letters should be guessed during evaluation
    guess_letters_qtty = 10
    # how often to display the results
    display_epoch = 100
    # Quantity of letters to predict by trained model
    num_to_predict = 100

    def __init__(self, vocab_size, batch_size=None, sequence_size=None,
                 dropout=None, mi_type=False):
        self.vocab_size = vocab_size
        # use mi type or not
        self.mi_type = mi_type
        if batch_size and sequence_size:
            self.batch_size = batch_size
            self.sequence_size = sequence_size
        if dropout:
            self.dropout = dropout

parser = argparse.ArgumentParser()
parser.add_argument('--enable_mi', dest="mi_type", action="store_true")
parser.add_argument('--disable_mi', dest="mi_type", action="store_false")
parser.set_defaults(mi_type=True)
args = parser.parse_args()

# Commence training and evaluation
# ================================
with tf.Graph().as_default(), tf.Session() as sess:
    if args.mi_type:
        logs_path = "logs/mi_type"
    else:
        logs_path = "logs/default_type"
    writer = tf.train.SummaryWriter(logs_path, sess.graph)
    # get our configs classes
    reader = DataReader(
        batch_size=Config.batch_size,
        sequence_size=Config.sequence_size,
        data_path='data/input_large.txt')
    reader.print_data_info()
    conf = train_conf = Config(
        vocab_size=reader.vocabularySize,
        mi_type=args.mi_type)
    eval_conf = Config(
        vocab_size=reader.vocabularySize,
        batch_size=1,
        sequence_size=1,
        dropout=1,
        mi_type=args.mi_type)

    with tf.variable_scope("SimpleRNN", reuse=None):
        train_model = RNN_Model(config=train_conf)
    with tf.variable_scope("SimpleRNN", reuse=True):
        eval_model = RNN_Model(config=eval_conf)

    tf.initialize_all_variables().run()

    mean_time_per_epoch = 0
    start_time = time.time()
    for epoch in range(conf.num_epochs):
        # additional variables to track the model
        losses = 0
        batch_index = 0

        # Model training
        current_learning_rate = conf.learning_rate * (conf.decay_rate ** epoch)
        sess.run(tf.assign(train_model.learning_rate, current_learning_rate))
        state = train_model.initial_state.eval()
        for batch_x, batch_y in reader.generateXYPairs():
            # run_metadata = tf.RunMetadata()
            feed_dict = {
                train_model.input_x: batch_x,
                train_model.input_y: batch_y,
                train_model.initial_state: state,
            }
            summary, perplexity, cost_res, loss_res, state, _ = sess.run(
                [train_model.merged, train_model.perplexity,
                 train_model.cost, train_model.loss,
                 train_model.final_state,
                 train_model.train_operation],
                feed_dict=feed_dict)
            losses += loss_res
            batch_index += 1
        writer.add_summary(summary, epoch)
        end_time = time.time()
        mean_time_per_epoch = (end_time - start_time) / (epoch + 1)

        # Model evaluation
        if epoch % conf.display_epoch == 0:
            mean_loss = np.mean(losses / (epoch + 1))
            status_message = ("{}/{} (epoch {}), cost_res: {:.3f}, "
                              "mean_loss: {:.5f}, perplexity: {:.3f}, "
                              "current learning rate: {:.5f}")
            print(status_message.format(
                      epoch * reader.total_batches + batch_index,
                      conf.num_epochs * reader.total_batches,
                      epoch, cost_res, mean_loss, perplexity,
                      current_learning_rate))
            estimated_seconds = mean_time_per_epoch * (conf.num_epochs - epoch)
            estimated_end = timedelta(seconds=estimated_seconds)
            print("Mean time per epoch: {:.5f}\n"
                  "Training should be finished in about: {}".format(
                    mean_time_per_epoch, estimated_end))

            predicted_text = eval_model.get_sample(
                sess=sess, chars=reader.unique_tokens,
                vocab=reader.token_to_id, num_to_predict=conf.num_to_predict,
                initial_sentence=' ')
            print(predicted_text)

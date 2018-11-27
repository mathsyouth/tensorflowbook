# TF code scaffolding for building simple models.

import tensorflow as tf

# initialize variables/model parameters

# define the training loop operations
def inference(X):
    # compute inference model over data X and return the result
    return

def loss(X, Y):
    # compute loss over training data X and expected values Y
    return

def inputs():
    # read/generate input training data X and expected outputs Y
    return

def train(total_loss):
    # train / adjust model parameters according to computed total loss
    return

def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    return

# Create a saver
saver = tf.train.Saver()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))
        if step % 1000 = 0:
            saver.save(sess, 'my-model', global_step=step)

    evaluate(sess, X, Y)
    
    # evaluation
    saver.save(sess, 'my-model', global_step=training_steps)

    coord.request_stop()
    coord.join(threads)
    sess.close()


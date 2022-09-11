import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



def loss_function(real, pred, loss_object):
                                                            # real = (m, Ty)
                                                            # pred = (m, Ty, num_tokens_target)

  mask = tf.math.logical_not(tf.math.equal(real, 0))        # want to select only non-zero values
                                                            # mask = (m, Ty), and is "True" for non-zero values

  loss_ = loss_object(real, pred)                           # compute loss for each time-step
                                                            # loss = (m, Ty)

  mask = tf.cast(mask, dtype=loss_.dtype)                   
  loss_ *= mask                                             # only count loss from non-zero values

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)           # divide sum(loss) by number of non-zero values



def accuracy_function(real, pred):                          # pred = (m, Ty, num_tokens_target)
  
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int32))      # accuracies = (m, Ty) -- binary values

  mask = tf.math.logical_not(tf.math.equal(real, 0))        # mask = (m, Ty) -- boolean values
  accuracies = tf.math.logical_and(mask, accuracies)        # suppress values where real value is 0

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)      # divide sum of 1s in "accuracies" by sum of 1s in "mask"


def compute_test_metrics(inp, tar, transformer, loss_object):
    # inp = (m, Tx)
    # tar = (m, Ty)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = transformer (inputs = (inp, tar_inp), training = False)
    test_loss = loss_function(tar_real, predictions, loss_object)
    test_acc = accuracy_function(tar_real, predictions)

    return test_loss, test_acc

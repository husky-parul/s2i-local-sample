#                                                            _
# tensorflowapp ds app
#
# (c) 2016 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

# import the Chris app superclass
from chrisapp.base import ChrisApp



class Tensorflowapp(ChrisApp):
    """
    test tf apps.
    """
    AUTHORS         = 'BillRainford (brain@redhat.com)'
    SELFPATH        = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC        = os.path.basename(__file__)
    EXECSHELL       = 'python3'
    TITLE           = 'tf training sample'
    CATEGORY        = 'tensorflow'
    TYPE            = 'ds'
    DESCRIPTION     = 'Sample Tensorflow training application plugin for ChRIS Project.'
    DOCUMENTATION   = 'http://wiki'
    VERSION         = '1'
    ICON            = '' # url of an icon image
    LICENSE         = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS = 1  # Override with integer value
    MAX_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT         = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT         = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Fill out this with key-value output descriptive info (such as an output file path
    # relative to the output dir) that you want to save to the output meta file when
    # called with the --saveoutputmeta flag
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        """
        self.add_argument('--prefix', dest='prefix', type=str, optional=False,
                          help='prefix for file names')
        self.add_argument('--inference_path', dest='inference_path', type=str,
                          optional=False, help='path of images')
        self.add_argument('--saved_model_name', dest='saved_model_name',
                          type=str, optional=False,
                          help='name for exporting saved model')

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        self.run_tensorflow_app(options)

    def run_tensorflow_app(self, options):
        digit_image = None
        if options.inference_path:
            str_path = os.path.abspath(options.inference_path)
            infer_image = Image.open(str_path)
            np_image = np.array(infer_image)
            np_image = np_image.flatten() / 255.0
            digit_image = np.reshape(np_image, (1, 784))
            print("Test Image shape: ", digit_image.shape)
        self.mnist_training(options, digit_image)

    def mnist_training(self, options, digit_image):
        print("Currently running as User ID: %s " % os.getuid())
        print("Trying to read from the directory %s " % options.inputdir)
        if os.path.isdir(options.inputdir):
            print("%s is a directory" % options.inputdir)
        else:
            print("%s is not a directory" % options.inputdir)
        if os.path.isdir((options.inputdir + "/data")):
            print("%s is a directory" % (options.inputdir + "/data"))
        else:
            print("%s is not a directory" % (options.inputdir + "/data"))
        print("Trying to read data from the directory %s " % (options.inputdir + "/data"))
        for path in os.listdir( (options.inputdir + "/data") ):
            print(path)
            if os.access( path, os.R_OK):
                print("  - readable")
            else:
                print("  - un-readable")
            if os.access( path, os.W_OK):
                print("  - writable")
            else:
                print("  - un-writeable")
            if os.access( path, os.X_OK):
                print("  - executable")
            else:
                print("  - non-executable")
        mnist = input_data.read_data_sets(options.inputdir + "/data", one_hot=True)
        image_size = 28
        labels_size = 10
        learning_rate = 0.05
        steps_number = 1000
        batch_size = 100
        # x is training data
        x = tf.placeholder(tf.float32, [None, image_size * image_size], name="myInput")
        labels = tf.placeholder(tf.float32, [None, labels_size])
        W = tf.Variable(tf.truncated_normal([image_size * image_size, labels_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
        # y is the output
        #y = tf.matmul(x, W) + b
        y = tf.nn.softmax(tf.matmul(x, W) + b, name="myOutput")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for i in range(steps_number):
            input_batch, labels_batch = mnist.train.next_batch(batch_size)
            feed_dict = {x: input_batch, labels: labels_batch}
            training_step.run(feed_dict=feed_dict)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("Step %d, training batch accuracy %g %%" % (i, train_accuracy * 100))
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels})
        acc = test_accuracy * 100
        print("Test accuracy: ", acc)
        self.create_output(options, "accuracy", acc)

        if digit_image is not None:
            prediction = sess.run(y, feed_dict={x: digit_image}).argmax()
            print("Inference value of test Image is : ", prediction)
            self.create_output(options, "inference", prediction)

        # Save the trained model
        str_outpath = os.path.join(options.outputdir, options.saved_model_name, self.VERSION)
        builder = tf.saved_model.builder.SavedModelBuilder(str_outpath)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()


    def create_output(self, options, key, value):
        new_name = options.prefix + key
        str_outpath = os.path.join(options.outputdir, new_name)
        str_outpath = os.path.abspath(str_outpath)
        print('Creating new file... %s' % str_outpath)
        if not os.path.exists(options.outputdir):
            try:
                os.mkdir(options.outputdir)
            except OSError:
                print("Creation of the directory %s failed" % options.outputdir)
            else:
                print("Successfully created the directory %s " % options.outputdir)
        with open(str_outpath, 'w') as f:
            f.write(str(value))



# ENTRYPOINT
if __name__ == "__main__":
    app = Tensorflowapp()
    app.launch()

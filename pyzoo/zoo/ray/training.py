import ray
import tensorflow as tf

# RayDataSet only containing essential meta data to retrieve data on remote 
class RayDataSet(object):
    def __init__(self, oid, batchsize):
       self.oid = oid
       self.batchsize = batchsize

    @staticmethod
    def from_rdd(rdd_inputs_targets, redis_addr, batchsize):
        # rdd_inputs_targets.
        # each partition should be a Array((inputs, targets))
        def fn(splitIndex, iterator):
            import ray
            import numpy as np
            # we need to make sure that we would not put 2 partitions in the same machine.
            ray.shutdown()
            ray.init(redis_addr)
            rows = iterator.next()
            oid = ray.put(rows)
            return oid
        list_of_oid = rdd_inputs_targets.mapPartitionsWithIndex(fn).collect()
        return [RayDataSet(oid, batchsize) for oid in list_of_oid]

    # inputs: tuple of ndarray
    # targets: tuple of ndarray
    def to_tf_dataset(self):
        import ray
        rows = ray.get(self.oid)
        # I suppose each row have the same shape
        self.features_placeholder = [tf.placeholder(tf.float32, row[0].shape) for row in rows]
        self.targets_placeholder = [tf.placeholder(tf.float32, row[1].shape) for row in rows]
        dataset = tf.data.Dataset.from_sparse_tensor_slices((self.features_placeholder, self.targets_placeholder))
        dataset = dataset.repeat.shuffle(1000).batch(self.batchsize)
        self.iterator = dataset.make_initializable_iterator()
        return dataset
    # sess.run(iterator.initializer, feed_dict={featuers_placeholder: featuers, targets_p:targets})



class Model(object):
    def __init__(self, keras_model):
        self.keras_model = keras_model
        # 1) start ps
        # 2) start worker

    # x = RayDataSet.from_rdd(rdd_inputs_targets, redis_addr, batchsize)
    def fit(self, x=None, batch_size=None, epoches=1, validation_data=None, workers=1):
        """
        x should be a dataset. (inputs, targets)
        validation_data is also a dataset
        :param batch_size:
        :param epoches:
        :param validation_data:
        :param workers:
        :return:
        """
    def _step(self):
        # 1) control worker and ps to run a step
        # 2) batch of data would autogen from woker, not trigger by driver

# ray.init()
# import numpy as np
# inputs = [np.random.randint(0, 1, (3, 4)), np.random.randint(0, 1, (3, 4))]
# targets = [inputs, inputs]
# oid = ray.put(targets)
# a = ray.get(oid)
# b = ray.get(oid)
# check ndarray.interl.base

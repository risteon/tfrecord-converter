#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import tensorflow as tf
import numpy as np

tf_major, _, _ = tf.__version__.split(".")
tf_major = int(tf_major)
if tf_major == 1 and not tf.executing_eagerly():
    raise RuntimeError("Need TensorFlow V1.x in eager execution mode.")
elif tf_major == 0:
    raise RuntimeError("TensorFlow V0.x is unsupported.")


class KittiSemanticsIterator:
    def __init__(
        self,
        velodyne_input_files,
        semantic_input_files,
        sequence_id="kitti_raw",
        performance=False,
    ):
        """Correspondence between velodyne data and semantics is given
        by sorted filenames.
        """

        if not self.is_sorted(velodyne_input_files):
            raise RuntimeError("Velodyne filenames are not in sorted order")
        if not self.is_sorted(semantic_input_files):
            raise RuntimeError("Semantic tfrecords filenames are not in sorted order")

        self.velodyne_input_files = velodyne_input_files
        self.record_dataset = tf.data.TFRecordDataset(
            tf.data.Dataset.from_tensor_slices(list(map(str, semantic_input_files))),
            compression_type="GZIP",
        )
        self.sequence_id = sequence_id

        def _parse(serialized_example_proto):
            # the keys are inherent to the defined dataset format
            parsed_features = tf.io.parse_single_example(
                serialized_example_proto,
                {
                    "probs_data": tf.io.FixedLenSequenceFeature(
                        shape=[], dtype=tf.float32, allow_missing=True
                    ),
                    "probs_shape": tf.io.FixedLenSequenceFeature(
                        shape=[], dtype=tf.int64, allow_missing=True
                    ),
                    "probs_mapping": tf.io.FixedLenSequenceFeature(
                        shape=[], dtype=tf.int64, allow_missing=True
                    ),
                },
            )
            return {
                "semantic_probabilities": parsed_features["probs_data"],
                "semantic_shape": parsed_features["probs_shape"],
                "semantic_mapping": parsed_features["probs_mapping"],
            }

        self.record_dataset = self.record_dataset.map(_parse, num_parallel_calls=6)
        if performance:
            self.record_dataset = self.record_dataset.map(
                self._calc_weights, num_parallel_calls=6
            )
        self.record_dataset = self.record_dataset.prefetch(buffer_size=8)
        # attach IDs from velodyne filenames
        self.zipped_data = zip(
            self.record_dataset,
            self.velodyne_input_files,
            [int(x.stem) for x in self.velodyne_input_files],
        )

    def __len__(self):
        return len(self.velodyne_input_files)

    def __iter__(self):
        return self.zipped_data.__iter__()

    def make_id_from_sample(self, sample):
        return "{}_{:04d}".format(self.sequence_id, sample[2])

    @staticmethod
    def jsd(p: tf.Tensor, q: tf.Tensor, base=np.e):
        """
            Implementation of pairwise Jensen-Shannon Divergence based on
            https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

            This returns NaNs for all zero probs (unlabeled).
        """
        import scipy.stats

        p, q = p.numpy(), q.numpy()
        # normalize p, q to probabilities
        p, q = p / p.sum(axis=-1, keepdims=True), q / q.sum(axis=-1, keepdims=True)
        p, q = p.transpose(), q.transpose()
        m = 1.0 / 2 * (p + q)
        jsd = (
            scipy.stats.entropy(p, m, base=base) / 2.0
            + scipy.stats.entropy(q, m, base=base) / 2.0
        )
        jsd = np.clip(jsd, 0.0, 1.0).transpose()
        return jsd

    @staticmethod
    def map_semantics_to_lima_v1_3_no_sky(tf_semantic_data):
        assert tf_semantic_data.shape.rank == 2  # < [P x C], points are padded

        num_classes_target = 12
        # for scatter operation, classes need to be the first dimension
        tf_semantic_data = tf.transpose(tf_semantic_data, perm=[1, 0])

        # specify the 'target' class for each source class
        # 'road (0) -> road(5)', 'sidewalk (1) -> sidewalk (6)', ...
        # now the target has a unlabeled class ([12]), where sky and traffic light get mapped to
        #                       0    1    2    3     4    5     6    7     8    9
        indices = tf.constant(
            [
                [5],
                [6],
                [1],
                [1],
                [12],
                [3],
                [12],
                [8],
                [10],
                [7],
                # 10   11   12    13   14   15   16   17   18
                [12],
                [2],
                [4],
                [11],
                [9],
                [9],
                [9],
                [0],
                [0],
            ]
        )
        tf_semantic_data = tf.scatter_nd(
            indices=indices,
            updates=tf_semantic_data,
            shape=tf.concat(
                ([num_classes_target + 1], tf.shape(tf_semantic_data)[1:]), axis=0
            ),
        )
        tf_semantic_data = tf.transpose(tf_semantic_data, perm=[1, 0])

        # remove the unlabeled class: decide where a unlabeled class is predicted
        # (using argmax) and set the corresponding row to zero. At the end, remove
        # the 12th unlabeled column
        tf_is_not_unlabeled = tf.not_equal(
            tf.math.argmax(tf_semantic_data, axis=-1, output_type=tf.dtypes.int32),
            num_classes_target,
        )
        tf_semantic_data = tf_semantic_data[..., :-1]
        tf_semantic_data /= tf.norm(tf_semantic_data, ord=1, axis=-1, keepdims=True)

        tf_semantic_data = tf.math.multiply(
            tf.expand_dims(
                tf.cast(tf_is_not_unlabeled, tf_semantic_data.dtype), axis=-1
            ),
            tf_semantic_data,
            name="set_unlabeled_rows_zero",
        )
        return tf_semantic_data

    @staticmethod
    def _calc_weights(tensors: {str, tf.Tensor}):
        tf_sem = tensors["semantic_probabilities"]
        tf_sem = tf.reshape(tf_sem, tensors["semantic_shape"])
        # get the number of input classes from the specified label mapping
        num_classes = 19
        tf_sem.set_shape((None, None, num_classes))

        # currently only stereo labeled data. Therefore there are two images
        tf_jsd = tf.py_function(
            KittiSemanticsIterator.jsd,
            inp=[tf_sem[0, :, :], tf_sem[1, :, :]],
            Tout=[tf.float32],
            name="calc_jsd",
        )[0]
        # [P]
        tf_jsd.set_shape(tf_sem.shape[-2:-1])
        # make distance from divergence (sqrt) and use as point-wise loss weight [B x P]
        tf_weight = 1.0 - tf.clip_by_value(
            tf.math.sqrt(tf_jsd), clip_value_min=0.0, clip_value_max=1.0
        )

        # simple prob merge over images [I x P x C] -> [P x C]
        tf_sem = tf.reduce_mean(tf_sem, axis=0)

        # map label set to reduced lidar set (if enabled)
        tf_sem = KittiSemanticsIterator.map_semantics_to_lima_v1_3_no_sky(tf_sem)

        tensors["semantic_probabilities"] = tf.reshape(tf_sem, [-1])
        tensors["semantic_shape"] = tf.cast(tf.shape(tf_sem), dtype=tf.int64)
        tensors["semantic_weights"] = tf_weight
        return tensors

    @staticmethod
    def reader(sample, sample_id):
        return {
            "sample_id": sample_id.encode("utf-8"),
            "point_cloud": KittiSemanticsIterator.read_pointcloud(sample[1]).flatten(),
            **sample[0],
        }

    @staticmethod
    def read_pointcloud(filepath):
        scan = np.fromfile(str(filepath), dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    @staticmethod
    def is_sorted(filelist):
        return all(filelist[i] <= filelist[i + 1] for i in range(len(filelist) - 1))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'


import tensorflow as tf


class Ohem(object):
    def __init__(self, logits, labels, train_mask,negative_ratio=3.):
        """
        :param logits: N*h*w*2
        :param labels: N*h*w*1
        :param train_mask:  N*h*w*1
        :param negative_ratio: float
        """
        self.logits = logits
        self.labels = labels
        self.negative_ratio = negative_ratio

        ### reshape
        cls_scores = tf.nn.softmax(self.logits)
        self.shape = tf.shape(self.logits)
        self.logits_flatten = tf.reshape(self.logits, shape=[self.shape[0], -1, self.shape[-1]])
        self.scores_flatten = tf.reshape(cls_scores, shape=[self.shape[0], -1, self.shape[-1]])

        labels_flatten = tf.reshape(self.labels, [self.shape[0], -1])
        self.train_mask= tf.cast(tf.reshape(train_mask, [self.shape[0], -1]), dtype=tf.float32)
        self.pos_class_label = tf.cast(labels_flatten*self.train_mask, dtype=tf.float32)
        #n_pos = tf.reduce_sum(tf.cast(self.pos_class_label, dtype=tf.float32))

    def ohem_loss(self, batch_size,label_weights=None):

        n_pos = tf.reduce_sum(tf.cast( self.pos_class_label, dtype=tf.float32))
        if label_weights is not None:
            weights_flatten = tf.reshape(label_weights, [self.shape[0], -1])
        else:
            label_weights=tf.ones_like(self.labels, dtype=tf.float32)
            weights_flatten = tf.reshape(label_weights, [self.shape[0], -1])

        def no_pos():
            return tf.constant(.0);

        def has_pos():
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_flatten,
                labels=tf.cast(self.pos_class_label,dtype=tf.int32))
            selected_neg_pixel_mask = self.ohem_batch_select(batch_size)

            weights = weights_flatten + tf.cast(selected_neg_pixel_mask, tf.float32)
            weights = weights*self.train_mask
            n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
            n_neg_pos =  tf.cast(tf.reduce_sum(weights), tf.float32)
            loss = tf.reduce_sum(loss * weights) / (n_pos + n_neg + 1)
            return loss

        return  tf.cond(n_pos > 0, has_pos, no_pos)

    @staticmethod
    def ohem_single_select(sin_predict, sin_target, sin_train_mask, negative_ratio=3.0):
        pos = (sin_target * sin_train_mask)
        neg = ((1 - sin_target) * sin_train_mask)
        neg_bool_mask = tf.cast(neg, dtype=tf.bool)

        n_pos = tf.reduce_sum(tf.cast(pos, dtype=tf.float32))

        def has_pos():
            return tf.cast(n_pos * negative_ratio, dtype=tf.int32)

        def no_pos():
            return tf.constant(10000, dtype=tf.int32)

        n_neg = tf.cond(n_pos > 0, has_pos, no_pos)

        n_neg = tf.minimum(tf.cast(tf.reduce_sum(tf.cast(neg, dtype=tf.float32)), dtype=tf.int32),
                           tf.cast(n_neg, dtype=tf.int32))

        def has_neg():
            neg_msk = tf.boolean_mask(sin_predict, neg_bool_mask)
            vals, _ = tf.nn.top_k(-neg_msk, k=n_neg)
            threshold = vals[-1]  # a negtive value
            select_neg_mask = tf.logical_and(neg_bool_mask, sin_predict <= -threshold)
            return select_neg_mask

        def no_neg():
            select_neg_mask = tf.zeros_like(neg_bool_mask)
            return select_neg_mask

        selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)

        return tf.cast(selected_neg_mask, tf.int32)

    def ohem_batch_select(self, Batch_size=1):
        selected_neg_mask = []
        neg_scores = self.scores_flatten[:, :, 0]
        for idx in range(int(Batch_size)):
            sin_predict = neg_scores[idx, :]
            sin_target = self.pos_class_label[idx, :]
            sin_train_mask = self.train_mask[idx, :]
            selected_mask = self.ohem_single_select(sin_predict,
                                                    sin_target,
                                                    sin_train_mask,
                                                    negative_ratio=self.negative_ratio)
            selected_neg_mask.append(selected_mask)

        selected_neg_mask = tf.stack(selected_neg_mask)

        return selected_neg_mask




class Ohem2(object):
    def __init__(self, logits, labels, train_masks, negative_ratio=3.):

        self.logits = logits
        self.labels = tf.cast(labels, dtype=tf.float32)
        self.train_masks = tf.cast(train_masks, dtype=tf.float32)
        self.negative_ratio = negative_ratio


    def sigmoid_ohem_loss(self, batch_size, label_weights=None):

        if label_weights is None:
            label_weights =tf.ones_like(self.labels)

        total_loss = 0
        for idx in range(int(batch_size)):
            logit = self.logits[idx, ...]
            label = self.labels[idx, ...]
            train_mask = self.train_masks[idx, ...]
            label_weight = label_weights[idx, ...]

            pos_mask = label * train_mask
            neg_mask = (1 - label) * train_mask
            n_pos = tf.reduce_sum(pos_mask)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(pos_mask, dtype=tf.int32),
                                                           logits=logit)

            def has_pos():
                return tf.cast(n_pos * self.negative_ratio, dtype=tf.int32)

            def no_pos():
                return tf.constant(10000, dtype=tf.int32)

            n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
            n_neg = tf.minimum(tf.cast(tf.reduce_sum(neg_mask), dtype=tf.int32), n_neg)

            loss_neg = loss * neg_mask * label_weight
            neg_bool_mask = tf.cast(neg_mask, dtype=tf.bool)
            vals, _ = tf.nn.top_k(tf.reshape(loss_neg, shape=[-1]), k=n_neg)
            selected_neg_mask = tf.logical_and(neg_bool_mask, loss_neg >= vals[-1])
            loss_mask = tf.cast(selected_neg_mask, dtype=tf.float32) + pos_mask

            loss_select = tf.reduce_sum(loss * loss_mask * label_weight)
            total_loss += loss_select / tf.reduce_sum(loss_mask * label_weight)

        return total_loss / batch_size

    def softmax_ohem_loss(self,batch_size, label_weights=None):

        if label_weights is None:
            label_weights = tf.ones_like(self.labels)

        total_loss = 0
        for idx in range(int(batch_size)):
            logit = self.logits[idx, ...]
            label = self.labels[idx, ...]
            train_mask = self.train_masks[idx, ...]
            label_weight = label_weights[idx, ...]

            pos_mask = label * train_mask
            neg_mask = (1 - label) * train_mask
            n_pos = tf.reduce_sum(pos_mask)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(pos_mask, dtype=tf.int32),
                                                                  logits=logit)

            def has_pos():
                return tf.cast(n_pos * self.negative_ratio, dtype=tf.int32)

            def no_pos():
                return tf.constant(10000, dtype=tf.int32)

            n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
            n_neg = tf.minimum(tf.cast(tf.reduce_sum(neg_mask), dtype=tf.int32), n_neg)

            loss_neg = loss*neg_mask*label_weight
            neg_bool_mask = tf.cast(neg_mask, dtype=tf.bool)
            vals, _ = tf.nn.top_k(tf.reshape(loss_neg, shape=[-1]), k=n_neg)
            selected_neg_mask = tf.logical_and(neg_bool_mask, loss_neg >= vals[-1])
            loss_mask = tf.cast(selected_neg_mask, dtype=tf.float32) + pos_mask

            loss_select = tf.reduce_sum(loss*loss_mask*label_weight)
            total_loss += loss_select/tf.reduce_sum(loss_mask*label_weight)

        return total_loss/batch_size



if __name__ == '__main__':
    # TODO ADD CODE
    pass

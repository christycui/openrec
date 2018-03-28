import tensorflow as tf
from openrec.modules.interactions import Interaction

class PointwiseMSERegularized(Interaction):

    """
    
    .. math::
        \min \sum_{ij}w_{j}(r_{ij} - u_i^T v_j)^2 + \lambda_u*||u||^2 + \lambda_focus*\sum_{j \in I}||v_j||^2 + \lambda_unfocus*\sum_{j \notin I }||v_j||^2
    
    where math:`u_i` denotes the representation for user :math:`i`; :math:`v_j` denotes representations for item; \
    :math:`I` denotes the set of items to focus on.
    
    Parameters
    ----------
    user: Tensorflow tensor
        Representations for users involved in the interactions. Shape: **[number of interactions, dimensionality of \
        user representations]**.
    item: Tensorflow tensor, required for testing
        Representations for items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    item_bias: Tensorflow tensor
        Biases for items involved in the interactions. Shape: **[number of interactions, 1]**.
    item_mask: Required for training.
        Representations for focused items (1) vs. unfocused item (0). Shape: **[number of interactions, 1]**.
    item_weights: Tensorflow tensor, required for training
        Weights for items involved in the interactions.
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    
    l2_reg_user: float, optional
        Weight for L2 regularization for users.
    l2_reg_focus: float, optional
        Weight for L2 regularization for focused items.
    l2_reg_unfocus: float, optional
        Weight for L2 regularization for unfocused items.
    train: bool, optional
        An indicator for training or serving phase.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. Beyond Globally Optimal: Focused Learning for Improved Recommendations
    """

    def __init__(self, user, item, item_bias, item_mask, item_weights=1.0, labels=None,
                l2_reg_user=30, l2_reg_focus=30, l2_reg_unfocus=60, train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        assert item is not None, 'item cannot be None'
        assert item_bias is not None, 'item bias cannot be None'
        assert item_mask is not None, 'item mask (focused vs. unfocused) cannot be None'
        self._user = user
        self._item = item
        self._item_bias = item_bias
        self._item_mask = item_mask

        if train:
            assert item_weights is not None, 'item weights cannot be None'
            self._item_weights = item_weights
            self._l2_reg_user = l2_reg_user
            self._l2_reg_focus = l2_reg_focus
            self._l2_reg_unfocus = l2_reg_unfocus
            self._labels = tf.reshape(tf.to_float(labels), (-1,))

        super(PointwiseMSERegularized, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            dot_user_item = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")
            predictions = dot_user_item + tf.reshape(self._item_bias, [-1])

            self._loss = tf.reduce_sum(self._item_weights*tf.nn.l2_loss(self._labels - predictions))\
                + self._l2_reg_user*tf.nn.l2_loss(self._user) \
                + self._l2_reg_focus*tf.nn.l2_loss(self._item_mask*self._item) \
                + self._l2_reg_unfocus*tf.nn.l2_loss((1-self._item_mask)*self._item)

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            prediction = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item") + tf.reshape(self._item_bias, [-1])

            self._outputs.append(prediction)


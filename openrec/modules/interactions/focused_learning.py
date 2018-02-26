import tensorflow as tf
from openrec.modules.interactions import Interaction

class FocusedLearning(Interaction):

    """
    
    .. math::
        \min \sum_{ij}w_{j}(r_{ij} - u_i^T v_j)^2 + \lambda_u*||u||^2 + \lambda_focus*\sum_{j}||v_j||^2 + \lambda_unfocus*\sum_{j}||v_j||^2

    
    Parameters
    ----------
    user: Tensorflow tensor
        Representations for users involved in the interactions. Shape: **[number of interactions, dimensionality of \
        user representations]**.
    item: Tensorflow tensor, required for testing
        Representations for items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    item_bias: Tensorflow tensor, required for testing
        Biases for items involved in the interactions. Shape: **[number of interactions, 1]**.
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    f_item: Tensorflow tensor, required for training
        Representations for focused items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    f_item_bias: Tensorflow tensor, required for training
        Biases for unfosued items involved in the interactions. Shape: **[number of interactions, 1]**.
    u_item: Tensorflow tensor, required for training
        Representations for negative items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    u_item_bias: Tensorflow tensor, required for training
        Biases for unfocused items involved in the interactions. Shape: **[number of interactions, 1]**.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
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

    def __init__(self, user, item=None, item_bias=None, f_item=None, f_item_bias=None, 
                u_item=None, u_item_bias=None, train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        self._user = user

        if train:
            assert f_item is not None, 'p_item cannot be None'
            assert u_item is not None, 'n_item cannot be None'
            assert f_item_bias is not None, 'p_item_bias cannot be None'
            assert u_item_bias is not None, 'n_item_bias cannot be None'

            self._f_item = f_item
            self._u_item = u_item
            self._f_item_bias = f_item_bias
            self._u_item_bias = u_item_bias
            self._labels = tf.reshape(tf.to_float(labels), (-1,))
        else:
            assert item is not None, 'item cannot be None'
            assert item_bias is not None, 'item_bias cannot be None'

            self._item = item
            self._item_bias = item_bias

        super(FocusedLearning, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            # TODO
            # need to define item_weights, l2_reg_user, l2_reg_focus, l2_reg_unfocus
            dot_user_item = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")
            # norm of user
            user_norm = tf.pow(tf.norm(self._user), 2)

            self._loss = tf.reduce_sum(item_weights*tf.pow(self._labels - predictions, 2) + l2_reg_user*user_norm) \
                + l2_reg_focus*tf.reduce_sum(tf.pow(tf.norm(self._f_item), 2)) \
                + l2_reg_unfocus*tf.reduce_sum(tf.pow(tf.norm(self._u_item), 2))

    def _build_serving_graph(self):

        pass


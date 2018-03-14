import tensorflow as tf
from openrec.modules.interactions import Interaction

class FocusedLearning(Interaction):

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
    item_weights: Tensorflow tensor, required for training
        Weights for items involved in the interactions.
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    f_item: Tensorflow tensor, required for training
        Representations for focused items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    u_item: Tensorflow tensor, required for training
        Representations for negative items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
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

    def __init__(self, user, item=None, item_weights=None, f_item=None, 
                u_item=None, l2_reg_user, l2_reg_focus, l2_reg_unfocus, train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        self._user = user

        if train:
            assert f_item is not None, 'p_item cannot be None'
            assert u_item is not None, 'n_item cannot be None'
            assert f_item is not None, 'f_item cannot be None'
            assert u_item is not None, 'u_item cannot be None'
            assert item_weights is not None, 'item_weights cannot be None'

            self._f_item = f_item
            self._u_item = u_item
            self._labels = tf.reshape(tf.to_float(labels), (-1,))
        else:
            assert item is not None, 'item cannot be None'

            self._item = item

        super(FocusedLearning, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            # TODO
            dot_user_item = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")
            # norm of user
            user_norm = tf.nn.l2_loss(self._user)

            self._loss = tf.reduce_sum(item_weights*tf.pow(self._labels - predictions, 2) + l2_reg_user*user_norm) \
                + l2_reg_focus*tf.reduce_sum(tf.nn.l2_loss(self._f_item)) \
                + l2_reg_unfocus*tf.reduce_sum(tf.nn.l2_loss(self._u_item))

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            prediction = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")
            
            self._outputs.append(prediction)


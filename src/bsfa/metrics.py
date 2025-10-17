import tensorflow as tf

def make_smape_metric_original(y_mu, y_sigma, eps=1e-6):
    y_mu_tf = tf.constant(y_mu, dtype=tf.float32)
    y_sigma_tf = tf.constant(y_sigma, dtype=tf.float32)
    eps_tf = tf.constant(eps, dtype=tf.float32)
    def smape(y_true, y_pred):
        yt = y_true * y_sigma_tf + y_mu_tf
        yp = y_pred * y_sigma_tf + y_mu_tf
        num = 2.0 * tf.abs(yp - yt)
        den = tf.abs(yt) + tf.abs(yp) + eps_tf
        return tf.reduce_mean(num / den)
    smape.__name__ = "smape"
    return smape

def _gather_features(t, idxs):
    if idxs is None:
        return t
    idxs_tf = tf.constant(idxs, dtype=tf.int32)
    if t.shape.rank == 1:
        return tf.gather(t, idxs_tf, axis=0)
    else:
        return tf.gather(t, idxs_tf, axis=1)

def make_masked_mape_metric_original(y_mu, y_sigma, denom_floor_vec, indices=None, eps=1e-6, name="masked_mape"):
    y_mu_tf = tf.constant(y_mu, dtype=tf.float32)
    y_sigma_tf = tf.constant(y_sigma, dtype=tf.float32)
    floor_tf = tf.constant(denom_floor_vec, dtype=tf.float32)
    eps_tf = tf.constant(eps, dtype=tf.float32)
    def masked_mape(y_true, y_pred):
        yt = y_true * y_sigma_tf + y_mu_tf
        yp = y_pred * y_sigma_tf + y_mu_tf
        yt = _gather_features(yt, indices)
        yp = _gather_features(yp, indices)
        floor_used = _gather_features(floor_tf, indices)
        den = tf.maximum(tf.abs(yt), floor_used) + eps_tf
        return tf.reduce_mean(tf.abs(yp - yt) / den)
    masked_mape.__name__ = name
    return masked_mape

def make_weighted_huber_plus_core_smape_loss(y_mu, y_sigma, feat_weights, core_idxs=(0,1,2),
                                             alpha=0.3, delta=1.0, eps=1e-6):
    y_mu_tf = tf.constant(y_mu, dtype=tf.float32)
    y_sigma_tf = tf.constant(y_sigma, dtype=tf.float32)
    w_tf = tf.constant(feat_weights, dtype=tf.float32)  # (F,)
    w_sum = tf.reduce_sum(w_tf)
    eps_tf = tf.constant(eps, dtype=tf.float32)
    delta_tf = tf.constant(delta, dtype=tf.float32)
    core_tf = tf.constant(list(core_idxs), dtype=tf.int32)

    def loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, delta_tf)
        lin = abs_err - quad
        huber = 0.5 * tf.square(quad) + delta_tf * lin
        huber_w = tf.reduce_sum(huber * w_tf, axis=1) / (w_sum + 1e-12)
        huber_w = tf.reduce_mean(huber_w)

        yt = y_true * y_sigma_tf + y_mu_tf
        yp = y_pred * y_sigma_tf + y_mu_tf
        yt_core = tf.gather(yt, core_tf, axis=1)
        yp_core = tf.gather(yp, core_tf, axis=1)
        num = 2.0 * tf.abs(yp_core - yt_core)
        den = tf.abs(yt_core) + tf.abs(yp_core) + eps_tf
        smape_core = tf.reduce_mean(num / den)
        return huber_w + alpha * smape_core
    loss.__name__ = "weighted_huber_plus_core_smape"
    return loss

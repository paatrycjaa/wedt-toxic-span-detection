from keras import backend as K
from keras import initializers as initializers, regularizers, constraints
from keras.engine.topology import Layer
import tensorflow as tf
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
@tf.keras.utils.register_keras_serializable()
# class attention(Layer):
#     def __init__(self,**kwargs):
#         super(attention,self).__init__(**kwargs)

#     def build(self,input_shape):
#         self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
#         self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
#         super(attention, self).build(input_shape)

#     def call(self,x):
#         et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
#         at=K.softmax(et)
#         at=K.expand_dims(at,axis=-1)
#         output=x*at
#         return K.sum(output,axis=1)

#     def compute_output_shape(self,input_shape):
#         return (input_shape[0],input_shape[-1])

#     def get_config(self):
#         return super(attention,self).get_config()
# class Attention(Layer):
#     def __init__(self,
#                  W_regularizer=None, u_regularizer=None, b_regularizer=None,
#                  W_constraint=None, u_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):

#         self.supports_masking = True
#         self.init = initializers.get('glorot_uniform')

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.u_regularizer = regularizers.get(u_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.u_constraint = constraints.get(u_constraint)
#         self.b_constraint = constraints.get(b_constraint)

#         self.bias = bias
#         super(Attention, self).__init__(**kwargs)


#     def call(self, x, mask=None):
#         uit = dot_product(x, self.W)

#         if self.bias:
#             uit += self.b

#         uit = K.tanh(uit)
#         ait = dot_product(uit, self.u)

#         a = K.exp(ait)

#         # apply mask after the exp. will be re-normalized next
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             a *= K.cast(mask, K.floatx())

#         # in some cases especially in the early stages of training the sum may be almost zero
#         # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
#         # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
#         a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

#         a = K.expand_dims(a)
#         weighted_input = x * a
#         return K.sum(weighted_input, axis=1)
    
#     def compute_mask(self, input, input_mask=None):
#         # do not pass the mask to the next layers
#         return None

#     def build(self, input_shape):
#         assert len(input_shape) == 3
#         self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)
#         if self.bias:
#             self.b = self.add_weight(shape=(input_shape[-1],),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)

#         self.u = self.add_weight(shape=(input_shape[-1],),
#                                  initializer=self.init,
#                                  name='{}_u'.format(self.name),
#                                  regularizer=self.u_regularizer,
#                                  constraint=self.u_constraint)

#         super(Attention, self).build(input_shape)
    
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[-1]
#     def get_config(self):
    
#         config = super().get_config().copy()
#         config.update({
#             'W_regularizer': self.W_regularizer,
#             'u_regularizer': self.u_regularizer,
#             'b_regularizer': self.b_regularizer,
#             'W_constraint': self.W_constraint,
#             'u_constraint': self.u_constraint,
#             'b_constraint': self.b_constraint,
#             'bias': self.bias
#         })
         
#         return config
class AttentionLayer(Layer):
    """
    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)
    - Yang et. al.
    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Theano backend
    """
    def __init__(self,attention_dim=100,return_coefficients=False,**kwargs):
        # Initializer 
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform') # initializes values with uniform distribution
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Builds all weights
        # W = Weight matrix, b = bias vector, u = context vector
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)),name='W')
        self.b = K.variable(self.init((self.attention_dim, )),name='b')
        self.u = K.variable(self.init((self.attention_dim, 1)),name='u')
        # self.trainable_weights = [self.W, self.b, self.u]

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, hit, mask=None):
        # Here, the actual calculation is done
        uit = K.bias_add(K.dot(hit, self.W),self.b)
        uit = K.tanh(uit)
        
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        
        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = hit * ait
        
        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), ait]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]
    def get_config(self):
    
        return super(AttentionLayer,self).get_config()
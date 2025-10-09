pytagi.nn.embedding
===================

.. py:module:: pytagi.nn.embedding


Classes
-------

.. autoapisummary::

   pytagi.nn.embedding.Embedding


Module Contents
---------------

.. py:class:: Embedding(num_embeddings: int, embedding_dim: int, input_size: int = 0, scale: float = 1.0, padding_idx: int = -1)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Embedding layer


   .. py:attribute:: num_embeddings


   .. py:attribute:: embedding_dim


   .. py:attribute:: input_size
      :value: 0



   .. py:attribute:: scale
      :value: 1.0



   .. py:attribute:: padding_idx
      :value: -1



   .. py:attribute:: _cpp_backend


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()

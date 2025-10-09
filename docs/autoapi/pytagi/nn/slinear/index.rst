pytagi.nn.slinear
=================

.. py:module:: pytagi.nn.slinear


Classes
-------

.. autoapisummary::

   pytagi.nn.slinear.SLinear


Module Contents
---------------

.. py:class:: SLinear(input_size: int, output_size: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoothering Linear layer


   .. py:attribute:: input_size


   .. py:attribute:: output_size


   .. py:attribute:: bias
      :value: True



   .. py:attribute:: gain_weight
      :value: 1.0



   .. py:attribute:: gain_bias
      :value: 1.0



   .. py:attribute:: init_method
      :value: 'He'



   .. py:attribute:: _cpp_backend


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()

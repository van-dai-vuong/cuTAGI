pytagi.nn.convtranspose2d
=========================

.. py:module:: pytagi.nn.convtranspose2d


Classes
-------

.. autoapisummary::

   pytagi.nn.convtranspose2d.ConvTranspose2d


Module Contents
---------------

.. py:class:: ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_type: int = 1, in_width: int = 0, in_height: int = 0, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Tranposed convolutional layer


   .. py:attribute:: in_channels


   .. py:attribute:: out_channels


   .. py:attribute:: kernel_size


   .. py:attribute:: is_bias
      :value: True



   .. py:attribute:: stride
      :value: 1



   .. py:attribute:: padding
      :value: 0



   .. py:attribute:: padding_type
      :value: 1



   .. py:attribute:: in_width
      :value: 0



   .. py:attribute:: in_height
      :value: 0



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

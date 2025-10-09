pytagi.nn.pooling
=================

.. py:module:: pytagi.nn.pooling


Classes
-------

.. autoapisummary::

   pytagi.nn.pooling.AvgPool2d
   pytagi.nn.pooling.MaxPool2d


Module Contents
---------------

.. py:class:: AvgPool2d(kernel_size: int, stride: int = -1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Average Pooling layer


   .. py:attribute:: kernel_size


   .. py:attribute:: stride
      :value: -1



   .. py:attribute:: padding
      :value: 0



   .. py:attribute:: padding_type
      :value: 0



   .. py:attribute:: _cpp_backend


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


.. py:class:: MaxPool2d(kernel_size: int, stride: int = 1, padding: int = 0, padding_type: int = 0)

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Max Pooling layer


   .. py:attribute:: kernel_size


   .. py:attribute:: stride
      :value: 1



   .. py:attribute:: padding
      :value: 0



   .. py:attribute:: padding_type
      :value: 0



   .. py:attribute:: _cpp_backend


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str

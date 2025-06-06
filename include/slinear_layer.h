#pragma once
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "linear_layer.h"

class SLinear : public Linear {
   public:
    SmoothSLinear smooth_states;
    int time_step = 0;

    SLinear(size_t ip_size, size_t op_size, bool bias = true,
            float gain_weight = 1.0f, float gain_bias = 1.0f,
            std::string method = "He", int time_step = 0, int device_idx = 0)
        : Linear(ip_size, op_size, bias, gain_weight, gain_bias, method,
                 device_idx),
          time_step(time_step) {}

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states, bool state_udapte) override;

    void smoother();
};

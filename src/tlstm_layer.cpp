
#include "../include/tlstm_layer.h"

#include <cmath>
#include <thread>
#include <tuple>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/lstm_layer.h"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
// OFFSET-AWARE FORWARD FREE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void tlstm_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        std::vector<float> &mu_a, std::vector<float> &var_a,
                        int start_chunk, int end_chunk, size_t input_size,
                        size_t output_size, int batch_size, int seq_len,
                        int time_step, bool bias, int w_pos, int b_pos,
                        std::vector<float> &mu_z, std::vector<float> &var_z) {
    for (int i = start_chunk; i < end_chunk; i++) {
        int row = i / output_size;  // batch index
        int col = i % output_size;  // output node index
        float sum_mu_z = 0.0f;
        float sum_var_z = 0.0f;
        int input_offset = row * seq_len * input_size + time_step * input_size;
        int output_offset =
            row * seq_len * output_size + time_step * output_size;

        for (int j = 0; j < input_size; j++) {
            float mu_a_tmp = mu_a[input_offset + j];
            float var_a_tmp = var_a[input_offset + j];
            float mu_w_tmp = mu_w[col * input_size + j + w_pos];
            float var_w_tmp = var_w[col * input_size + j + w_pos];

            sum_mu_z += mu_w_tmp * mu_a_tmp;
            sum_var_z += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                         var_w_tmp * mu_a_tmp * mu_a_tmp;
        }
        if (bias) {
            mu_z[output_offset + col] = sum_mu_z + mu_b[row + b_pos];
            var_z[output_offset + col] = sum_var_z + var_b[row + b_pos];
        } else {
            mu_z[output_offset + col] = sum_mu_z;
            var_z[output_offset + col] = sum_var_z;
        }
    }
}

void tlstm_cat_activations_and_prev_states(std::vector<float> &a,
                                           std::vector<float> &b, int n, int m,
                                           int batch_size, int x_offset,
                                           int input_offset,
                                           std::vector<float> &c) {
    for (int k = 0; k < batch_size; k++) {
        for (int i = 0; i < n; i++) {
            c[input_offset + i + k * (n + m)] = a[x_offset + i + k * n];
        }
        for (int j = 0; j < m; j++) {
            c[input_offset + j + n + k * (n + m)] = b[j + k * m];
        }
    }
}

void tlstm_cov_input_cell_states(std::vector<float> &var_ha,
                                 std::vector<float> &mu_w,
                                 std::vector<float> &jcb_i_ga,
                                 std::vector<float> &jcb_c_ga, int w_pos_i,
                                 int w_pos_c, int ni, int no, int batch_size,
                                 int seq_len, int time_step,
                                 std::vector<float> &cov_i_c) {
    float sum;
    int k, i, m_idx;
    for (int x = 0; x < batch_size; x++) {
        for (int z = 0; z < no; z++) {
            sum = 0;
            for (int j = 0; j < ni + no; j++) {
                k = j + z * (ni + no);
                m_idx = j + time_step * (ni + no) + x * (seq_len * (ni + no));
                sum += mu_w[w_pos_i + k] * var_ha[m_idx] * mu_w[w_pos_c + k];
            }
            i = z + time_step * no + x * seq_len * no;
            cov_i_c[i] = jcb_i_ga[i] * sum * jcb_c_ga[i];
        }
    }
}

void tlstm_cell_state_mean_var(
    std::vector<float> &mu_f_ga, std::vector<float> &var_f_ga,
    std::vector<float> &mu_i_ga, std::vector<float> &var_i_ga,
    std::vector<float> &mu_c_ga, std::vector<float> &var_c_ga,
    std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev,
    std::vector<float> &cov_i_c, int no, int batch_size, int seq_len,
    int time_step, std::vector<float> &mu_c, std::vector<float> &var_c) {
    int k;
    for (int x = 0; x < batch_size; x++) {
        for (int z = 0; z < no; z++) {
            k = z + time_step * no + x * no * seq_len;
            mu_c[k] = mu_f_ga[k] * mu_c_prev[k] + mu_i_ga[k] * mu_c_ga[k] +
                      cov_i_c[k];
            var_c[k] = var_c_prev[k] * mu_f_ga[k] * mu_f_ga[k] +
                       var_c_prev[k] * var_f_ga[k] +
                       var_f_ga[k] * mu_c_prev[k] * mu_c_prev[k] +
                       var_c_ga[k] * mu_i_ga[k] * mu_i_ga[k] +
                       var_i_ga[k] * var_c_ga[k] +
                       var_i_ga[k] * mu_c_ga[k] * mu_c_ga[k] +
                       powf(cov_i_c[k], 2) +
                       2 * cov_i_c[k] * mu_i_ga[k] * mu_c_ga[k];
        }
    }
}

void tlstm_cov_output_tanh_cell_states(
    std::vector<float> &mu_w, std::vector<float> &var_ha,
    std::vector<float> &mu_c_prev, std::vector<float> &jcb_ca,
    std::vector<float> &jcb_f_ga, std::vector<float> &mu_i_ga,
    std::vector<float> &jcb_i_ga, std::vector<float> &mu_c_ga,
    std::vector<float> &jcb_c_ga, std::vector<float> &jcb_o_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int batch_size,
    int seq_len, int time_step, std::vector<float> &cov_tanh_c) {
    float sum_fo, sum_io, sum_oc;
    int k, m, i;
    for (int x = 0; x < batch_size; x++) {
        for (int z = 0; z < no; z++) {
            sum_fo = 0.0f;
            sum_io = 0.0f;
            sum_oc = 0.0f;
            for (int j = 0; j < ni; j++) {
                k = j + z * (ni + no);
                m = j + time_step * (ni + no) + x * (seq_len * (ni + no));
                sum_fo += mu_w[w_pos_f + k] * var_ha[m] * mu_w[w_pos_o + k];
                sum_io += mu_w[w_pos_i + k] * var_ha[m] * mu_w[w_pos_o + k];
                sum_oc += mu_w[w_pos_c + k] * var_ha[m] * mu_w[w_pos_o + k];
            }
            i = z + time_step * no + x * no * seq_len;
            cov_tanh_c[i] =
                jcb_ca[i] * (jcb_o_ga[i] * sum_fo * jcb_f_ga[i] * mu_c_prev[i] +
                             jcb_o_ga[i] * sum_io * jcb_i_ga[i] * mu_c_ga[i] +
                             jcb_o_ga[i] * sum_oc * jcb_c_ga[i] * mu_i_ga[i]);
        }
    }
}

void tlstm_hidden_state_mean_var(
    std::vector<float> &mu_o_ga, std::vector<float> &var_o_ga,
    std::vector<float> &mu_ca, std::vector<float> &var_ca,
    std::vector<float> &cov_o_tanh_c, int no, int batch_size, int seq_len,
    int time_step, std::vector<float> &mu_z, std::vector<float> &var_z) {
    int k;
    for (int x = 0; x < batch_size; x++) {
        for (int z = 0; z < no; z++) {
            k = z + time_step * no + x * no * seq_len;
            mu_z[k] = mu_o_ga[k] * mu_ca[k] + cov_o_tanh_c[k];
            var_z[k] =
                var_ca[k] * mu_o_ga[k] * mu_o_ga[k] + var_ca[k] * var_o_ga[k] +
                var_o_ga[k] * mu_ca[k] * mu_ca[k] + powf(cov_o_tanh_c[k], 2) +
                2 * cov_o_tanh_c[k] * mu_o_ga[k] * mu_ca[k];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// OFFSET-AWARE BACKWARD FREE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void tlstm_delta_mean_var_z(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_mu_out,
    std::vector<float> &delta_var_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int batch_size, int seq_len, int time_step,
    std::vector<float> &delta_mu, std::vector<float> &delta_var) {
    int ni_c = ni + no;
    for (int x = 0; x < batch_size; x++) {
        for (int z = 0; z < ni; z++) {
            float sum_mf = 0, sum_mi = 0, sum_mc = 0, sum_mo = 0;
            float sum_var_z = 0;
            for (int j = 0; j < no; j++) {
                int k = j + x * seq_len * no + time_step * no;

                // Forget gate
                float Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                              mw[ni_c * j + z + w_pos_f] * mc_prev[k];

                // Input gate
                float Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                              mw[ni_c * j + z + w_pos_i] * mc_ga[k];

                // Cell state gate
                float Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                              mw[ni_c * j + z + w_pos_c] * mi_ga[k];

                // Output gate
                float Czz_o = Jo_ga[k] * mw[ni_c * j + z + w_pos_o] * mca[k];

                sum_mf += Czz_f * delta_mu_out[k];
                sum_mi += Czz_i * delta_mu_out[k];
                sum_mc += Czz_c * delta_mu_out[k];
                sum_mo += Czz_o * delta_mu_out[k];
                sum_var_z +=
                    powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) * delta_var_out[k];
            }
            int m = x * ni + z;
            delta_mu[m] = sum_mf + sum_mi + sum_mc + sum_mo;
            delta_var[m] = sum_var_z;
        }
    }
}

void tlstm_delta_mean_var_w(
    std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_mu,
    std::vector<float> &delta_var, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int batch_size, int seq_len, int time_step,
    std::vector<float> &sum_mu_w_f, std::vector<float> &sum_var_w_f,
    std::vector<float> &sum_mu_w_i, std::vector<float> &sum_var_w_i,
    std::vector<float> &sum_mu_w_c, std::vector<float> &sum_var_w_c,
    std::vector<float> &sum_mu_w_o, std::vector<float> &sum_var_w_o) {
    int ni_c = ni + no;
    for (int row = 0; row < ni_c; row++) {
        for (int col = 0; col < no; col++) {
            float s_mu_f = 0, s_var_f = 0, s_mu_i = 0, s_var_i = 0;
            float s_mu_c = 0, s_var_c = 0, s_mu_o = 0, s_var_o = 0;
            for (int x = 0; x < batch_size; x++) {
                int k = col + x * seq_len * no + time_step * no;
                int l = row + x * seq_len * ni_c + time_step * ni_c;

                float Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
                s_mu_f += Cwa_f * delta_mu[k];
                s_var_f += Cwa_f * delta_var[k] * Cwa_f;

                float Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
                s_mu_i += Cwa_i * delta_mu[k];
                s_var_i += Cwa_i * delta_var[k] * Cwa_i;

                float Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
                s_mu_c += Cwa_c * delta_mu[k];
                s_var_c += Cwa_c * delta_var[k] * Cwa_c;

                float Cwa_o = Jo_ga[k] * mca[k] * mha[l];
                s_mu_o += Cwa_o * delta_mu[k];
                s_var_o += Cwa_o * delta_var[k] * Cwa_o;
            }
            int m = col * ni_c + row;
            sum_mu_w_f[m] += s_mu_f;
            sum_var_w_f[m] += s_var_f;
            sum_mu_w_i[m] += s_mu_i;
            sum_var_w_i[m] += s_var_i;
            sum_mu_w_c[m] += s_mu_c;
            sum_var_w_c[m] += s_var_c;
            sum_mu_w_o[m] += s_mu_o;
            sum_var_w_o[m] += s_var_o;
        }
    }
}

void tlstm_delta_mean_var_b(
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
    std::vector<float> &Jo_ga, std::vector<float> &mc_prev,
    std::vector<float> &mca, std::vector<float> &Jc,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int no,
    int batch_size, int seq_len, int time_step, std::vector<float> &sum_mu_b_f,
    std::vector<float> &sum_var_b_f, std::vector<float> &sum_mu_b_i,
    std::vector<float> &sum_var_b_i, std::vector<float> &sum_mu_b_c,
    std::vector<float> &sum_var_b_c, std::vector<float> &sum_mu_b_o,
    std::vector<float> &sum_var_b_o) {
    for (int row = 0; row < no; row++) {
        float s_mu_f = 0, s_var_f = 0, s_mu_i = 0, s_var_i = 0;
        float s_mu_c = 0, s_var_c = 0, s_mu_o = 0, s_var_o = 0;
        for (int x = 0; x < batch_size; x++) {
            int k = row + x * seq_len * no + time_step * no;

            float Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
            s_mu_f += Cwa_f * delta_mu[k];
            s_var_f += Cwa_f * delta_var[k] * Cwa_f;

            float Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
            s_mu_i += Cwa_i * delta_mu[k];
            s_var_i += Cwa_i * delta_var[k] * Cwa_i;

            float Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
            s_mu_c += Cwa_c * delta_mu[k];
            s_var_c += Cwa_c * delta_var[k] * Cwa_c;

            float Cwa_o = Jo_ga[k] * mca[k];
            s_mu_o += Cwa_o * delta_mu[k];
            s_var_o += Cwa_o * delta_var[k] * Cwa_o;
        }
        sum_mu_b_f[row] += s_mu_f;
        sum_var_b_f[row] += s_var_f;
        sum_mu_b_i[row] += s_mu_i;
        sum_var_b_i[row] += s_var_i;
        sum_mu_b_c[row] += s_mu_c;
        sum_var_b_c[row] += s_var_c;
        sum_mu_b_o[row] += s_mu_o;
        sum_var_b_o[row] += s_var_o;
    }
}

////////////////////////////////////////////////////////////////////////////////
// TLSTM CLASS
////////////////////////////////////////////////////////////////////////////////

TLSTM::TLSTM(size_t input_size, size_t output_size, int seq_len, bool bias,
             float gain_w, float gain_b, std::string init_method,
             int device_idx)
    : gain_w(gain_w), gain_b(gain_b), init_method(init_method) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->seq_len = seq_len;
    this->bias = bias;
    this->device_idx = device_idx;

    this->get_number_param();
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
}

TLSTM::~TLSTM() {}

std::string TLSTM::get_layer_info() const {
    return "TLSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string TLSTM::get_layer_name() const { return "TLSTM"; }

LayerType TLSTM::get_layer_type() const { return LayerType::TLSTM; }

int TLSTM::get_input_size() { return this->input_size * this->seq_len; }

int TLSTM::get_output_size() { return this->output_size * this->seq_len; }

int TLSTM::get_max_num_states() {
    int in_size = static_cast<int>(this->input_size) * this->seq_len;
    int out_size = static_cast<int>(this->output_size) * this->seq_len;
    return std::max(in_size, out_size);
}

void TLSTM::get_number_param() {
    this->num_weights =
        4 * this->output_size * (this->input_size + this->output_size);
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = 4 * this->output_size;
        this->b_pos_f = 0;
        this->b_pos_i = this->output_size;
        this->b_pos_c = 2 * this->output_size;
        this->b_pos_o = 3 * this->output_size;
    }

    this->w_pos_f = 0;
    this->w_pos_i = this->output_size * (this->input_size + this->output_size);
    this->w_pos_c =
        2 * this->output_size * (this->input_size + this->output_size);
    this->w_pos_o =
        3 * this->output_size * (this->input_size + this->output_size);
}

void TLSTM::init_weight_bias() {
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_lstm(this->init_method, this->gain_w, this->gain_b,
                              this->input_size, this->output_size,
                              this->num_weights, this->num_biases);
}

void TLSTM::forward(BaseHiddenStates &input_states,
                    BaseHiddenStates &output_states,
                    BaseTempStates &temp_states) {
    if (this->input_size * this->seq_len != input_states.actual_size) {
        std::string message = "Input size mismatch: " +
                              std::to_string(this->input_size * this->seq_len) +
                              " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;
    int T = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;

    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_states.set_num_states(batch_size * T * no,
                                         batch_size * T * ni);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.seq_len = this->seq_len;
    output_states.actual_size = this->output_size;

    int state_size = batch_size * no;
    std::vector<float> h_prev_mu(state_size);
    std::vector<float> h_prev_var(state_size);
    std::vector<float> c_prev_mu(state_size);
    std::vector<float> c_prev_var(state_size);

    for (int i = 0; i < state_size; i++) {
        h_prev_mu[i] = this->lstm_states.mu_h_prior[i];
        h_prev_var[i] = this->lstm_states.var_h_prior[i];
        c_prev_mu[i] = this->lstm_states.mu_c_prior[i];
        c_prev_var[i] = this->lstm_states.var_c_prior[i];
    }

    int end_chunk = no * batch_size;

    for (int t = 0; t < T; t++) {
        int state_off = t * state_size;
        int input_off = t * batch_size * ni_c;
        int x_off = t * batch_size * ni;

        for (int i = 0; i < state_size; i++) {
            lstm_states.mu_h_prev[state_off + i] = h_prev_mu[i];
            lstm_states.var_h_prev[state_off + i] = h_prev_var[i];
            lstm_states.mu_c_prev[state_off + i] = c_prev_mu[i];
            lstm_states.var_c_prev[state_off + i] = c_prev_var[i];
        }

        tlstm_cat_activations_and_prev_states(input_states.mu_a, h_prev_mu, ni,
                                              no, batch_size, x_off, input_off,
                                              lstm_states.mu_ha);
        tlstm_cat_activations_and_prev_states(input_states.var_a, h_prev_var,
                                              ni, no, batch_size, x_off,
                                              input_off, lstm_states.var_ha);

        tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                           lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                           ni_c, no, batch_size, this->seq_len, t, this->bias,
                           this->w_pos_f, this->b_pos_f, lstm_states.mu_f_ga,
                           lstm_states.var_f_ga);
        sigmoid_mean_var(lstm_states.mu_f_ga, lstm_states.var_f_ga, state_off,
                         state_off + state_size, lstm_states.mu_f_ga,
                         lstm_states.jcb_f_ga, lstm_states.var_f_ga);

        tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                           lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                           ni_c, no, batch_size, this->seq_len, t, this->bias,
                           this->w_pos_i, this->b_pos_i, lstm_states.mu_i_ga,
                           lstm_states.var_i_ga);
        sigmoid_mean_var(lstm_states.mu_i_ga, lstm_states.var_i_ga, state_off,
                         state_off + state_size, lstm_states.mu_i_ga,
                         lstm_states.jcb_i_ga, lstm_states.var_i_ga);

        tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                           lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                           ni_c, no, batch_size, this->seq_len, t, this->bias,
                           this->w_pos_c, this->b_pos_c, lstm_states.mu_c_ga,
                           lstm_states.var_c_ga);
        tanh_mean_var(lstm_states.mu_c_ga, lstm_states.var_c_ga, state_off,
                      state_off + state_size, lstm_states.mu_c_ga,
                      lstm_states.jcb_c_ga, lstm_states.var_c_ga);

        tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                           lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                           ni_c, no, batch_size, this->seq_len, t, this->bias,
                           this->w_pos_o, this->b_pos_o, lstm_states.mu_o_ga,
                           lstm_states.var_o_ga);
        sigmoid_mean_var(lstm_states.mu_o_ga, lstm_states.var_o_ga, state_off,
                         state_off + state_size, lstm_states.mu_o_ga,
                         lstm_states.jcb_o_ga, lstm_states.var_o_ga);

        tlstm_cov_input_cell_states(
            lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
            lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c, ni, no,
            batch_size, this->seq_len, t, lstm_states.cov_i_c);

        // TODO: need to review c_prev_mu as this function consider c_prev_mu =
        // batch size  * seq_len * no
        tlstm_cell_state_mean_var(
            lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
            lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
            c_prev_mu, c_prev_var, lstm_states.cov_i_c, no, batch_size,
            this->seq_len, t, lstm_states.mu_c, lstm_states.var_c);

        tanh_mean_var(lstm_states.mu_c, lstm_states.var_c, state_off,
                      state_off + state_size, lstm_states.mu_ca,
                      lstm_states.jcb_ca, lstm_states.var_ca);

        tlstm_cov_output_tanh_cell_states(
            this->mu_w, lstm_states.var_ha, c_prev_mu, lstm_states.jcb_ca,
            lstm_states.jcb_f_ga, lstm_states.mu_i_ga, lstm_states.jcb_i_ga,
            lstm_states.mu_c_ga, lstm_states.jcb_c_ga, lstm_states.jcb_o_ga,
            this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o, ni, no,
            batch_size, this->seq_len, t, lstm_states.cov_o_tanh_c);

        tlstm_hidden_state_mean_var(
            lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
            lstm_states.var_ca, lstm_states.cov_o_tanh_c, no, batch_size,
            this->seq_len, t, output_states.mu_a, output_states.var_a);

        for (int i = 0; i < state_size; i++) {
            h_prev_mu[i] = output_states.mu_a[state_off + i];
            h_prev_var[i] = output_states.var_a[state_off + i];
            c_prev_mu[i] = lstm_states.mu_c[state_off + i];
            c_prev_var[i] = lstm_states.var_c[state_off + i];
        }
    }

    for (int i = 0; i < state_size; i++) {
        this->lstm_states.mu_h_prior[i] = h_prev_mu[i];
        this->lstm_states.var_h_prior[i] = h_prev_var[i];
        this->lstm_states.mu_c_prior[i] = c_prev_mu[i];
        this->lstm_states.var_c_prior[i] = c_prev_var[i];
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void TLSTM::backward(BaseDeltaStates &input_delta_states,
                     BaseDeltaStates &output_delta_states,
                     BaseTempStates &temp_states, bool state_udapte) {
    int batch_size = input_delta_states.block_size;
    int T = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;
    int state_size = batch_size * no;

    // Recurrent deltas from t+1, initialized to zero
    std::vector<float> delta_rec_mu(state_size, 0.0f);
    std::vector<float> delta_rec_var(state_size, 0.0f);

    // Temp buffer for full [x(t), h(t-1)] deltas
    std::vector<float> delta_full_mu(batch_size * ni_c, 0.0f);
    std::vector<float> delta_full_var(batch_size * ni_c, 0.0f);

    // Combined delta (direct + recurrent)
    std::vector<float> combined_delta_mu(state_size);
    std::vector<float> combined_delta_var(state_size);

    // Accumulators for weight and bias deltas (raw sums, no Sw/Sb yet)
    int w_size = ni_c * no;
    std::vector<float> sum_mw_f(w_size, 0.0f), sum_Sw_f(w_size, 0.0f);
    std::vector<float> sum_mw_i(w_size, 0.0f), sum_Sw_i(w_size, 0.0f);
    std::vector<float> sum_mw_c(w_size, 0.0f), sum_Sw_c(w_size, 0.0f);
    std::vector<float> sum_mw_o(w_size, 0.0f), sum_Sw_o(w_size, 0.0f);

    std::vector<float> sum_mb_f(no, 0.0f), sum_Sb_f(no, 0.0f);
    std::vector<float> sum_mb_i(no, 0.0f), sum_Sb_i(no, 0.0f);
    std::vector<float> sum_mb_c(no, 0.0f), sum_Sb_c(no, 0.0f);
    std::vector<float> sum_mb_o(no, 0.0f), sum_Sb_o(no, 0.0f);

    // We need a temporary buffer for combined deltas stored in the same layout
    // as lstm_states (to pass to delta_z, delta_w, delta_b functions).
    // We reuse input_delta_states for the combined output delta.
    // But we need to be careful: input_delta_states.delta_mu/var has size
    // B * T * no. We'll write the combined delta at each timestep's slot.
    std::vector<float> &delta_mu_buf = input_delta_states.delta_mu;
    std::vector<float> &delta_var_buf = input_delta_states.delta_var;

    for (int t = T - 1; t >= 0; t--) {
        int state_off = t * state_size;
        int input_off = t * batch_size * ni_c;

        // Combine: delta_h(t) = direct_delta[t] + delta_recurrent
        for (int i = 0; i < state_size; i++) {
            combined_delta_mu[i] =
                delta_mu_buf[state_off + i] + delta_rec_mu[i];
            combined_delta_var[i] =
                delta_var_buf[state_off + i] + delta_rec_var[i];
        }

        // Write combined delta back so delta functions can read from state_off
        for (int i = 0; i < state_size; i++) {
            delta_mu_buf[state_off + i] = combined_delta_mu[i];
            delta_var_buf[state_off + i] = combined_delta_var[i];
        }

        if (state_udapte) {
            // Compute delta for full [x(t), h(t-1)]
            std::fill(delta_full_mu.begin(), delta_full_mu.end(), 0.0f);
            std::fill(delta_full_var.begin(), delta_full_var.end(), 0.0f);

            tlstm_delta_mean_var_z(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                delta_mu_buf, delta_var_buf, this->w_pos_f, this->w_pos_i,
                this->w_pos_c, this->w_pos_o, no, ni, batch_size, T, t,
                delta_full_mu, delta_full_var);

            // Split: first ni per sample -> output_delta (for previous layer)
            //        last no per sample  -> delta_rec (for next backward iter)
            int x_off = t * batch_size * ni;
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < ni; j++) {
                    output_delta_states.delta_mu[x_off + b * ni + j] =
                        delta_full_mu[b * ni_c + j];
                    output_delta_states.delta_var[x_off + b * ni + j] =
                        delta_full_var[b * ni_c + j];
                }
                for (int j = 0; j < no; j++) {
                    delta_rec_mu[b * no + j] = delta_full_mu[b * ni_c + ni + j];
                    delta_rec_var[b * no + j] =
                        delta_full_var[b * ni_c + ni + j];
                }
            }
        }

        if (param_update) {
            // Accumulate weight deltas
            tlstm_delta_mean_var_w(
                lstm_states.mu_ha, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                delta_mu_buf, delta_var_buf, this->w_pos_f, this->w_pos_i,
                this->w_pos_c, this->w_pos_o, no, ni, batch_size, T, t,
                sum_mw_f, sum_Sw_f, sum_mw_i, sum_Sw_i, sum_mw_c, sum_Sw_c,
                sum_mw_o, sum_Sw_o);

            if (this->bias) {
                tlstm_delta_mean_var_b(
                    lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                    lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                    lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                    lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                    lstm_states.mu_ca, lstm_states.jcb_ca, delta_mu_buf,
                    delta_var_buf, no, batch_size, T, t, sum_mb_f, sum_Sb_f,
                    sum_mb_i, sum_Sb_i, sum_mb_c, sum_Sb_c, sum_mb_o, sum_Sb_o);
            }
        }
    }

    // Multiply accumulated sums by Sw/Sb to get final deltas
    if (param_update) {
        for (int m = 0; m < w_size; m++) {
            this->delta_mu_w[m + this->w_pos_f] =
                sum_mw_f[m] * this->var_w[m + this->w_pos_f];
            this->delta_var_w[m + this->w_pos_f] =
                this->var_w[m + this->w_pos_f] * sum_Sw_f[m] *
                this->var_w[m + this->w_pos_f];

            this->delta_mu_w[m + this->w_pos_i] =
                sum_mw_i[m] * this->var_w[m + this->w_pos_i];
            this->delta_var_w[m + this->w_pos_i] =
                this->var_w[m + this->w_pos_i] * sum_Sw_i[m] *
                this->var_w[m + this->w_pos_i];

            this->delta_mu_w[m + this->w_pos_c] =
                sum_mw_c[m] * this->var_w[m + this->w_pos_c];
            this->delta_var_w[m + this->w_pos_c] =
                this->var_w[m + this->w_pos_c] * sum_Sw_c[m] *
                this->var_w[m + this->w_pos_c];

            this->delta_mu_w[m + this->w_pos_o] =
                sum_mw_o[m] * this->var_w[m + this->w_pos_o];
            this->delta_var_w[m + this->w_pos_o] =
                this->var_w[m + this->w_pos_o] * sum_Sw_o[m] *
                this->var_w[m + this->w_pos_o];
        }

        if (this->bias) {
            for (int r = 0; r < no; r++) {
                this->delta_mu_b[r + this->b_pos_f] =
                    sum_mb_f[r] * this->var_b[r + this->b_pos_f];
                this->delta_var_b[r + this->b_pos_f] =
                    this->var_b[r + this->b_pos_f] * sum_Sb_f[r] *
                    this->var_b[r + this->b_pos_f];

                this->delta_mu_b[r + this->b_pos_i] =
                    sum_mb_i[r] * this->var_b[r + this->b_pos_i];
                this->delta_var_b[r + this->b_pos_i] =
                    this->var_b[r + this->b_pos_i] * sum_Sb_i[r] *
                    this->var_b[r + this->b_pos_i];

                this->delta_mu_b[r + this->b_pos_c] =
                    sum_mb_c[r] * this->var_b[r + this->b_pos_c];
                this->delta_var_b[r + this->b_pos_c] =
                    this->var_b[r + this->b_pos_c] * sum_Sb_c[r] *
                    this->var_b[r + this->b_pos_c];

                this->delta_mu_b[r + this->b_pos_o] =
                    sum_mb_o[r] * this->var_b[r + this->b_pos_o];
                this->delta_var_b[r + this->b_pos_o] =
                    this->var_b[r + this->b_pos_o] * sum_Sb_o[r] *
                    this->var_b[r + this->b_pos_o];
            }
        }
    }

    // Update priors using the recurrent delta that reached t=0
    // delta_rec now contains the delta for h at t=-1 (unused by previous layer,
    // but used to update the hidden state prior)
    lstm_update_prev_hidden_states_worker(
        this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
        delta_rec_mu, delta_rec_var, 0, state_size,
        this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior);

    // For cell state prior update, we need the delta at the output (t=0)
    // and the jcb_ca, mu_o_ga at t=0
    lstm_update_prev_cell_states_worker(
        this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
        this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga, combined_delta_mu,
        combined_delta_var, 0, state_size, this->lstm_states.mu_c_prior,
        this->lstm_states.var_c_prior);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
TLSTM::get_LSTM_states() const {
    return std::make_tuple(lstm_states.mu_h_prior, lstm_states.var_h_prior,
                           lstm_states.mu_c_prior, lstm_states.var_c_prior);
}

void TLSTM::set_LSTM_states(const std::vector<float> &mu_h,
                            const std::vector<float> &var_h,
                            const std::vector<float> &mu_c,
                            const std::vector<float> &var_c) {
    this->lstm_states.mu_h_prior = mu_h;
    this->lstm_states.var_h_prior = var_h;
    this->lstm_states.mu_c_prior = mu_c;
    this->lstm_states.var_c_prior = var_c;

    this->lstm_states.mu_h_prev = mu_h;
    this->lstm_states.var_h_prev = var_h;
    this->lstm_states.mu_c_prev = mu_c;
    this->lstm_states.var_c_prev = var_c;
}

void TLSTM::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }
    if (this->training) {
        this->allocate_param_delta();
    }
}

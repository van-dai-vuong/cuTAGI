#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "../../include/param_init.h"
#include "../../include/tagi_autocov.h"

using namespace tagi_autocov;

namespace {

float truth(float x) { return std::sin(3.0f * x) + 0.3f * x * x; }

// The user only defines the forward pass -- 2-layer MLP, mirrors the
// tagi_autocov.py demo's MLP class. Reproducibility comes from the
// process-wide SeedManager (same convention as the rest of the library,
// e.g. pytagi.manual_seed()) -- Linear no longer takes a per-instance seed.
class MLP : public Module {
   public:
    explicit MLP(unsigned seed) {
        SeedManager::get_instance().set_seed(seed);
        fc1 = std::make_shared<Linear>(1, 64);
        fc2 = std::make_shared<Linear>(64, 1);
        register_module(fc1);
        register_module(fc2);
    }

    TensorPtr forward(const TensorPtr& x) override {
        return fc2->forward(relu(fc1->forward(x)));
    }

   private:
    std::shared_ptr<Linear> fc1, fc2;
};

// Arbitrary graph: skip connection + element-wise multiply (gating). The
// backward inference follows whatever graph forward() builds -- no
// hand-written backward() needed even for this non-sequential topology.
class GatedResNet : public Module {
   public:
    explicit GatedResNet(unsigned seed) {
        SeedManager::get_instance().set_seed(seed);
        inp = std::make_shared<Linear>(1, 32);
        h = std::make_shared<Linear>(32, 32);
        gate = std::make_shared<Linear>(32, 32);
        out = std::make_shared<Linear>(32, 1);
        register_module(inp);
        register_module(h);
        register_module(gate);
        register_module(out);
    }

    TensorPtr forward(const TensorPtr& x) override {
        TensorPtr a = relu(inp->forward(x));
        TensorPtr branch =
            mul(tanh_act(h->forward(a)), sigmoid(gate->forward(a)));
        return out->forward(add(a, branch));
    }

   private:
    std::shared_ptr<Linear> inp, h, gate, out;
};

struct Dataset {
    std::vector<float> x;
    std::vector<float> y;
};

Dataset make_dataset(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(-2.0f, 2.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    Dataset d;
    d.x.resize(n);
    d.y.resize(n);
    for (int i = 0; i < n; ++i) {
        d.x[i] = unif(rng);
        d.y[i] = truth(d.x[i]) + noise(rng);
    }
    return d;
}

float train_rmse(Module& net, const Dataset& d) {
    float sq_err = 0.0f;
    for (size_t i = 0; i < d.x.size(); ++i) {
        TensorPtr out = net.forward(tensor({d.x[i]}, {1, 1}));
        float diff = out->mu[0] - d.y[i];
        sq_err += diff * diff;
    }
    return std::sqrt(sq_err / static_cast<float>(d.x.size()));
}

void train_one_epoch(Module& net, const Dataset& d, float var_v,
                     std::mt19937& rng) {
    std::vector<int> order(d.x.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::shuffle(order.begin(), order.end(), rng);

    for (int i : order) {
        TensorPtr out = net.forward(tensor({d.x[i]}, {1, 1}));
        out->observe({d.y[i]}, var_v);  // forward built the graph; this is
                                        // the entire "backward pass" call
    }
}

}  // namespace

class TagiAutocovTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TagiAutocovTest, MlpOnlyForwardDefinedConverges) {
    Dataset train = make_dataset(150, /*seed=*/42);
    MLP net(/*seed=*/1);
    float sigma_v = 0.1f;

    std::mt19937 rng(7);
    float rmse_before = train_rmse(net, train);
    for (int epoch = 0; epoch < 40; ++epoch) {
        train_one_epoch(net, train, sigma_v * sigma_v, rng);
    }
    float rmse_after = train_rmse(net, train);

    EXPECT_LT(rmse_after, rmse_before);
    EXPECT_LT(rmse_after, 0.5f);
}

TEST_F(TagiAutocovTest, ArbitraryGraphSkipAndGateConverges) {
    // Verifies the automatic backward sweep correctly threads gains through
    // a non-sequential graph (skip-add and an element-wise GMA product),
    // not just a linear stack of layers.
    Dataset train = make_dataset(150, /*seed=*/42);
    GatedResNet net(/*seed=*/3);
    float sigma_v = 0.1f;

    std::mt19937 rng(11);
    float rmse_before = train_rmse(net, train);
    for (int epoch = 0; epoch < 40; ++epoch) {
        train_one_epoch(net, train, sigma_v * sigma_v, rng);
    }
    float rmse_after = train_rmse(net, train);

    EXPECT_LT(rmse_after, rmse_before);
    EXPECT_LT(rmse_after, 0.5f);
}

TEST_F(TagiAutocovTest, ObserveTwiceOnSameGraphThrows) {
    MLP net(5);
    TensorPtr out = net.forward(tensor({0.3f}, {1, 1}));
    out->observe({0.1f}, 0.01f);
    EXPECT_THROW(out->observe({0.1f}, 0.01f), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Covariance tape: automatic cross-covariance between nodes that share
// stochastic ancestry (the LSTM i_t/c_tilde_t situation).
// ---------------------------------------------------------------------------

namespace {

ParamPtr make_param(std::vector<float> mu, std::vector<float> var,
                    std::vector<int> shape, const std::string& name) {
    return std::make_shared<Parameter>(std::move(mu), std::move(var),
                                       std::move(shape), name);
}

}  // namespace

// mul(sigmoid(W1 x), tanh(W2 x)) -- both factors read the same x, so
// cov(a, b) = jcb_a (sum_k W1_k var_x_k W2_k) jcb_b must appear in the
// product's moments, exactly the structure of the production
// lstm_cov_input_cell_states (src/lstm_layer.cpp).
TEST_F(TagiAutocovTest, TapeSuppliesCrossCovOfGatingProduct) {
    clear_cov_tapes();
    // 1 output, 2 inputs: everything hand-computable.
    auto w1 = make_param({0.5f, -0.3f}, {0.02f, 0.02f}, {1, 2}, "W1");
    auto w2 = make_param({0.8f, 0.4f}, {0.03f, 0.03f}, {1, 2}, "W2");
    std::vector<float> mu_x = {0.7f, -0.2f}, var_x = {0.5f, 0.25f};
    TensorPtr x = tensor(mu_x, {1, 2}, var_x);
    x->track_cov();

    TensorPtr a = sigmoid(linear(x, w1));
    TensorPtr b = tanh_act(linear(x, w2));
    TensorPtr p = mul(a, b);

    // Hand computation (same moment formulas as the ops themselves).
    float mz1 = 0.5f * 0.7f + (-0.3f) * (-0.2f);
    float mz2 = 0.8f * 0.7f + 0.4f * (-0.2f);
    auto lin_var = [&](const std::vector<float>& mw,
                       const std::vector<float>& vw) {
        float v = 0.0f;
        for (int k = 0; k < 2; ++k) {
            v += vw[k] * (var_x[k] + mu_x[k] * mu_x[k]) +
                 mw[k] * mw[k] * var_x[k];
        }
        return v;
    };
    float vz1 = lin_var({0.5f, -0.3f}, {0.02f, 0.02f});
    float vz2 = lin_var({0.8f, 0.4f}, {0.03f, 0.03f});
    float ma = 1.0f / (1.0f + std::exp(-mz1));
    float ja = ma * (1.0f - ma), va = ja * ja * vz1;
    float mb = std::tanh(mz2);
    float jb = 1.0f - mb * mb, vb = jb * jb * vz2;
    // cov through the shared root x: weights are independent, so only
    // mu_W enters (matching lstm_cov_input_cell_states).
    float rho = ja * (0.5f * var_x[0] * 0.8f + (-0.3f) * var_x[1] * 0.4f) * jb;
    float mu_exp = ma * mb + rho;
    float var_exp = va * vb + va * mb * mb + vb * ma * ma + rho * rho +
                    2.0f * rho * ma * mb;

    EXPECT_NE(rho, 0.0f);
    EXPECT_NEAR(p->mu[0], mu_exp, 1e-6f);
    EXPECT_NEAR(p->var[0], var_exp, 1e-6f);

    // Same graph WITHOUT track_cov: the old independent moments.
    clear_cov_tapes();
    TensorPtr x2 = tensor(mu_x, {1, 2}, var_x);
    TensorPtr p2 = mul(sigmoid(linear(x2, w1)), tanh_act(linear(x2, w2)));
    EXPECT_NEAR(p2->mu[0], ma * mb, 1e-6f);
    EXPECT_NEAR(p2->var[0], va * vb + va * mb * mb + vb * ma * ma, 1e-6f);
}

// With deterministic weights, W1 x + W2 x must be EXACTLY (W1 + W2) x once
// the tape supplies cov(W1 x, W2 x) -- a strong end-to-end identity that
// the independent treatment misses by 2 cov.
TEST_F(TagiAutocovTest, TapeMakesAddOfSharedInputLinearsExact) {
    clear_cov_tapes();
    auto w1 = make_param({0.5f, -0.3f, 0.1f}, {0.0f, 0.0f, 0.0f}, {1, 3}, "W1");
    auto w2 = make_param({0.8f, 0.4f, -0.6f}, {0.0f, 0.0f, 0.0f}, {1, 3}, "W2");
    auto ws = make_param({1.3f, 0.1f, -0.5f}, {0.0f, 0.0f, 0.0f}, {1, 3}, "Ws");
    std::vector<float> mu_x = {0.7f, -0.2f, 1.1f}, var_x = {0.5f, 0.25f, 0.8f};

    TensorPtr x = tensor(mu_x, {1, 3}, var_x);
    x->track_cov();
    TensorPtr s = add(linear(x, w1), linear(x, w2));

    TensorPtr xr = tensor(mu_x, {1, 3}, var_x);
    TensorPtr ref = linear(xr, ws);

    EXPECT_NEAR(s->mu[0], ref->mu[0], 1e-5f);
    EXPECT_NEAR(s->var[0], ref->var[0], 1e-5f);
}

// A full LSTM-style step trains at least as well with the tape on; here we
// only assert the mechanics: the graph with tracked roots still conditions
// correctly (backward runs, parameters move toward the observation).
TEST_F(TagiAutocovTest, TrackedGraphStillConditionsCorrectly) {
    clear_cov_tapes();
    SeedManager::get_instance().set_seed(9);
    auto gate1 = std::make_shared<Linear>(2, 4);
    auto gate2 = std::make_shared<Linear>(2, 4);
    auto head = std::make_shared<Linear>(4, 1);

    auto forward = [&](float x0, float x1) {
        TensorPtr x = tensor({x0, x1}, {1, 2}, {0.1f, 0.1f});
        clear_cov_tapes();
        x->track_cov();
        TensorPtr gated =
            mul(sigmoid(gate1->forward(x)), tanh_act(gate2->forward(x)));
        return head->forward(gated);
    };

    TensorPtr out = forward(0.4f, -0.9f);
    float before = std::abs(out->mu[0] - 1.5f);
    for (int it = 0; it < 20; ++it) {
        forward(0.4f, -0.9f)->observe({1.5f}, 0.01f);
    }
    float after = std::abs(forward(0.4f, -0.9f)->mu[0] - 1.5f);
    EXPECT_LT(after, before);
    EXPECT_LT(after, 0.1f);
    clear_cov_tapes();
}

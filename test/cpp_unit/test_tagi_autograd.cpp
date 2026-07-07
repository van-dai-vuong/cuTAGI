#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "../../include/param_init.h"
#include "../../include/tagi_autograd.h"

using namespace tagi_autograd;

namespace {

float truth(float x) { return std::sin(3.0f * x) + 0.3f * x * x; }

// The user only defines the forward pass -- 2-layer MLP, mirrors the
// tagi_autograd.py demo's MLP class. Reproducibility comes from the
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

class TagiAutogradTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TagiAutogradTest, MlpOnlyForwardDefinedConverges) {
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

TEST_F(TagiAutogradTest, ArbitraryGraphSkipAndGateConverges) {
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

TEST_F(TagiAutogradTest, ObserveTwiceOnSameGraphThrows) {
    MLP net(5);
    TensorPtr out = net.forward(tensor({0.3f}, {1, 1}));
    out->observe({0.1f}, 0.01f);
    EXPECT_THROW(out->observe({0.1f}, 0.01f), std::runtime_error);
}

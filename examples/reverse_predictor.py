import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import (
    Embedding,
    Linear,
    MultiheadAttention,
    OutputUpdater,
    RMSNorm,
    Sequential,
)


class ReverseDataset:
    """Generates random sequences and their reversed versions."""

    def __init__(self, vocab_size: int = 10, seq_len: int = 16):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def next_batch(self, batch_size: int):
        x = np.random.randint(self.vocab_size, size=(batch_size, self.seq_len))
        x = x.reshape(batch_size, self.seq_len, 1).astype(np.float32)
        y = np.flip(x, axis=1)
        return x, y.reshape(-1).astype(np.int32)


def main(
    num_epochs: int = 50,
    batch_size: int = 2,
    seq_len: int = 5,
    vocab_size: int = 8,
    embed_dim: int = 64,
    num_heads: int = 4,
    sigma_v: float = 1.0,
    steps_per_epoch: int = 100,
):
    """Train a TAGI attention model on the sequence reversal task."""
    task = ReverseDataset(vocab_size=vocab_size, seq_len=seq_len)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=vocab_size)

    hrc = utils.get_hierarchical_softmax(vocab_size)
    hrc_class_len = hrc.len

    net = Sequential(
        Embedding(vocab_size, embed_dim, input_size=seq_len),
        MultiheadAttention(
            embed_dim,
            num_heads,
            num_heads,
            seq_len=seq_len,
            bias=False,
            init_method="Xavier",
        ),
        RMSNorm([embed_dim]),
        Linear(embed_dim, hrc_class_len),
    )

    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,),
        sigma_v**2,
        dtype=np.float32,
    )

    out_updater = OutputUpdater(net.device)

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        net.train()
        error_rates = []
        for _ in range(steps_per_epoch):
            x, y = task.next_batch(batch_size)
            m_pred, v_pred = net(x)

            y_obs, y_idx, _ = utils.label_to_obs(
                labels=y, num_classes=vocab_size
            )
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y_obs,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            error_rate = metric.error_rate(m_pred, v_pred, y)
            error_rates.append(error_rate)

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | error: {avg_error * 100:.2f}%"
        )

    x_test, y_test = task.next_batch(batch_size)
    net.eval()
    m_pred, v_pred = net(x_test)
    predicted = metric.get_predicted_labels(m_pred, v_pred)

    x_test = x_test.reshape(batch_size, seq_len).astype(int)
    y_test = y_test.reshape(batch_size, seq_len)
    predicted = predicted.reshape(batch_size, seq_len)

    num_show = min(5, batch_size)
    print(f"\nTest Results (showing {num_show} of {batch_size}):")
    for i in range(num_show):
        print(f"  Input:      {x_test[i].tolist()}")
        print(f"  Target:     {y_test[i].tolist()}")
        print(f"  Prediction: {predicted[i].tolist()}")
        print()

    accuracy = np.mean(predicted == y_test)
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    fire.Fire(main)

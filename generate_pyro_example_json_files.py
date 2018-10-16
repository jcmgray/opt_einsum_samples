from __future__ import division, print_function

import itertools
import json

from opt_einsum import get_symbol, contract_expression


def save(name, eq, shapes):
    filename = 'oe_sample_{}.json'.format(name)
    print('saving {}'.format(filename))
    with open(filename, 'wb') as f:
        json.dump({'eq': eq, 'shapes': shapes}, f)


def symbol_stream():
    for i in itertools.count():
        yield get_symbol(i)


# Examples of batched Hidden Markov Model inference.
#
#  x ---> x ---> x      ---->
#  |      |      |      time
#  V      V      V
#  y      y      y
def make_hmm_example(length, latent_dim=32, observed_dim=4, batch_dim=10, query=None):
    symbols = symbol_stream()
    b = next(symbols)
    shapes = []
    inputs = []
    xs = []
    ys = []
    for t in range(length):
        xs.append(next(symbols))
        ys.append(next(symbols))

        # Add observation matrix.
        shapes.append([batch_dim, latent_dim, observed_dim])
        inputs.append(b + xs[-1] + ys[-1])

        if t >= 1:
            # Add transition matrix.
            shapes.append([batch_dim, latent_dim, latent_dim])
            inputs.append(b + xs[-2] + xs[-1])

    inputs = ','.join(inputs)
    output = '' if query is None else b + xs[query]
    eq = inputs + '->' + output
    name = 'hmm_total_{}_{}_{}_{}_{}'.format(
        length, latent_dim, observed_dim, batch_dim, 'total' if query is None else query)
    contract_expression(eq, *shapes, optimize='eager')  # smoke test
    save(name, eq, shapes)


# Examples of batched Dynamic Bayes Net inference.
#
#         w
#       / | \
#      /  |  \
#    /    |    \
#  x ---> x ---> x
#  |      |      |
#  y ---> y ---> y          ---->
#  |      |      |          time
#  V      V      V
#  z      z      z
def make_dbn_example(length, global_dim=2, latent_dim=32, observed_dim=4, batch_dim=10, query=None):
    symbols = symbol_stream()
    b = next(symbols)
    w = next(symbols)
    shapes = []
    inputs = []
    xs = []
    ys = []
    zs = []
    for t in range(length):
        xs.append(next(symbols))
        ys.append(next(symbols))
        zs.append(next(symbols))

        # Add vertical dependencies.
        shapes.append([batch_dim, global_dim, latent_dim])
        inputs.append(b + w + xs[-1])

        shapes.append([batch_dim, latent_dim, latent_dim])
        inputs.append(b + xs[-1] + ys[-1])

        shapes.append([batch_dim, latent_dim, latent_dim])
        inputs.append(b + ys[-1] + zs[-1])

        if t >= 1:
            # Add horizontal dependencies.
            shapes.append([batch_dim, latent_dim, latent_dim])
            inputs.append(b + xs[-2] + xs[-1])

            shapes.append([batch_dim, latent_dim, latent_dim])
            inputs.append(b + ys[-2] + ys[-1])

    inputs = ','.join(inputs)
    output = '' if query is None else b + w + xs[query] + ys[query]
    eq = inputs + '->' + output
    name = 'dbn_total_{}_{}_{}_{}_{}'.format(
        length, latent_dim, observed_dim, batch_dim, 'total' if query is None else query)
    contract_expression(eq, *shapes, optimize='eager')  # smoke test
    save(name, eq, shapes)


def make_all():
    for length in [10, 100]:
        for query in [None, 0, -1, length // 2]:
            make_hmm_example(length, query=query)
            make_dbn_example(length, query=query)


if __name__ == '__main__':
    make_all()

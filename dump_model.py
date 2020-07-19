#!/usr/bin/env python3

import os
import sys
import json

import tensorflow as tf

assert tf.__version__.split(".")[0] == "1", "You must use Tensorflow 1.x to dump the parameters"

class PropObject:
  def __init__(*args, **kwargs):
    args[0]._obj_props = {**{k: v for d in args[1:] for k, v in d.items()}, **kwargs}
    args[0].__dict__.update(args[0]._obj_props)

model_dir = sys.argv[1]

with open(os.path.join(model_dir, "hparams.json"), "rt") as f:
  hparams = PropObject(json.load(f))

header = PropObject(n_vocab=hparams.n_vocab,
                    n_context=hparams.n_ctx,
                    n_features=hparams.n_embd,
                    n_heads=hparams.n_head,
                    n_layers=hparams.n_layer,
                    n_attn_features=hparams.n_embd//hparams.n_head,
                    n_neurons=hparams.n_embd*4)

header_text = "\n".join([f"#define {k} ({v})" for k, v in header._obj_props.items()])
with open("hyperparameters.h", "wt") as f:
  f.write(f"""
#pragma once
#ifndef __HYPERPARAMS_H__
#define __HYPERPARAMS_H__
{header_text}
#endif
""")

with tf.Session(graph=tf.Graph()) as sess:
  with tf.variable_scope("model"):
    wpe = tf.get_variable("wpe", [hparams.n_ctx, hparams.n_embd])
    wte = tf.get_variable("wte", [hparams.n_vocab, hparams.n_embd])

    ln_1_g = [None] * hparams.n_layer
    ln_1_b = [None] * hparams.n_layer
    c_attn_w = [None] * hparams.n_layer
    c_attn_b = [None] * hparams.n_layer
    c_attn_proj_w = [None] * hparams.n_layer
    c_attn_proj_b = [None] * hparams.n_layer
    ln_2_g = [None] * hparams.n_layer
    ln_2_b = [None] * hparams.n_layer
    c_fc_w = [None] * hparams.n_layer
    c_fc_b = [None] * hparams.n_layer
    c_mlp_proj_w = [None] * hparams.n_layer
    c_mlp_proj_b = [None] * hparams.n_layer

    for layer in range(0, hparams.n_layer):
      with tf.variable_scope(f"h{layer}"):
        with tf.variable_scope("ln_1"):
          ln_1_g[layer] = tf.get_variable("g", [hparams.n_embd])
          ln_1_b[layer] = tf.get_variable("b", [hparams.n_embd])
        with tf.variable_scope("attn"):
          with tf.variable_scope("c_attn"):
            c_attn_w[layer] = tf.get_variable("w", [1, hparams.n_embd, hparams.n_embd * 3])
            c_attn_b[layer] = tf.get_variable("b", [hparams.n_embd * 3])
          with tf.variable_scope("c_proj"):
            c_attn_proj_w[layer] = tf.get_variable("w", [1, hparams.n_embd, hparams.n_embd])
            c_attn_proj_b[layer] = tf.get_variable("b", [hparams.n_embd])
        with tf.variable_scope("ln_2"):
          ln_2_g[layer] = tf.get_variable("g", [hparams.n_embd])
          ln_2_b[layer] = tf.get_variable("b", [hparams.n_embd])
        with tf.variable_scope("mlp"):
          with tf.variable_scope("c_fc"):
            c_fc_w[layer] = tf.get_variable("w", [1, hparams.n_embd, hparams.n_embd * 4])
            c_fc_b[layer] = tf.get_variable("b", [hparams.n_embd * 4])
          with tf.variable_scope("c_proj"):
            c_mlp_proj_w[layer] = tf.get_variable("w", [1, hparams.n_embd * 4, hparams.n_embd])
            c_mlp_proj_b[layer] = tf.get_variable("b", [hparams.n_embd])

    with tf.variable_scope("ln_f"):
      ln_f_g = tf.get_variable("g", [hparams.n_embd])
      ln_f_b = tf.get_variable("b", [hparams.n_embd])

  token_embeddings = wte
  position_embeddings = wpe

  attn_input_transform = tf.stack(ln_1_g, axis=0)
  attn_input_translate = tf.stack(ln_1_b, axis=0)

  attn_transforms, = tf.unstack(tf.stack(c_attn_w, axis=0), axis=1)

  attn_query_transform, attn_key_transform, attn_value_transform = tf.unstack(
    tf.transpose(tf.reshape(attn_transforms, list(attn_transforms.shape[:-1]) + [3, hparams.n_head, -1]),
                 [2, 0, 3, 4, 1]), axis=0)
  attn_query_translate, attn_key_translate, attn_value_translate = tf.unstack(
    tf.reshape(tf.stack(c_attn_b, axis=0), [hparams.n_layer, 3, hparams.n_head, -1]), axis=1)

  attn_output_transform = tf.transpose(tf.reshape(tf.stack(c_attn_proj_w, axis=0),
                                                  [hparams.n_layer, hparams.n_head, -1, hparams.n_embd]),
                                       [0, 3, 1, 2])
  attn_output_translate = tf.stack(c_attn_proj_b, axis=0)

  neural_input_transform = tf.stack(ln_2_g, axis=0)
  neural_input_translate = tf.stack(ln_2_b, axis=0)

  neural_activations_transform, = tf.unstack(tf.transpose(tf.stack(c_fc_w, axis=0), [0, 1, 3, 2]), axis=1)
  neural_activations_translate = tf.stack(c_fc_b, axis=0)

  neural_output_transform, = tf.unstack(tf.transpose(tf.stack(c_mlp_proj_w, axis=0), [0, 1, 3, 2]), axis=1)
  neural_output_translate = tf.stack(c_mlp_proj_b, axis=0)

  finish_transform = ln_f_g
  finish_translate = ln_f_b

  assert token_embeddings.shape == [header.n_vocab, header.n_features]
  assert position_embeddings.shape == [header.n_context, header.n_features]

  assert attn_input_transform.shape == [header.n_layers, header.n_features]
  assert attn_input_translate.shape == [header.n_layers, header.n_features]

  assert attn_query_transform.shape == [header.n_layers, header.n_heads, header.n_attn_features, header.n_features]
  assert attn_query_translate.shape == [header.n_layers, header.n_heads, header.n_attn_features]
  assert attn_key_transform.shape == [header.n_layers, header.n_heads, header.n_attn_features, header.n_features]
  assert attn_key_translate.shape == [header.n_layers, header.n_heads, header.n_attn_features]
  assert attn_value_transform.shape == [header.n_layers, header.n_heads, header.n_attn_features, header.n_features]
  assert attn_value_translate.shape == [header.n_layers, header.n_heads, header.n_attn_features]

  assert attn_output_transform.shape == [header.n_layers, header.n_features, header.n_heads, header.n_attn_features]
  assert attn_output_translate.shape == [header.n_layers, header.n_features]

  assert neural_input_transform.shape == [header.n_layers, header.n_features]
  assert neural_input_translate.shape == [header.n_layers, header.n_features]

  assert neural_activations_transform.shape == [header.n_layers, header.n_neurons, header.n_features]
  assert neural_activations_translate.shape == [header.n_layers, header.n_neurons]

  assert neural_output_transform.shape == [header.n_layers, header.n_features, header.n_neurons]
  assert neural_output_translate.shape == [header.n_layers, header.n_features]

  assert finish_transform.shape == [header.n_features]
  assert finish_translate.shape == [header.n_features]

  output_tensor = tf.concat(list(map(lambda t: tf.reshape(t, [-1]), [
    token_embeddings, position_embeddings,
    attn_input_transform, attn_input_translate,
    attn_query_transform, attn_query_translate, attn_key_transform, attn_key_translate, attn_value_transform, attn_value_translate,
    attn_output_transform, attn_output_translate,
    neural_input_transform, neural_input_translate,
    neural_activations_transform, neural_activations_translate,
    neural_output_transform, neural_output_translate,
    finish_transform, finish_translate
  ])), axis=0)

  saver = tf.train.Saver()
  ckpt = tf.train.latest_checkpoint(model_dir)
  saver.restore(sess, ckpt)

  output = sess.run(output_tensor)

with open("model.bin", "wb") as f:
  output.tofile(f)

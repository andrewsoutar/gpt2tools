#include "gpt2.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <math.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "hyperparameters.h"

struct model_parameters {
  float token_embeddings[n_vocab][n_features];
  float position_embeddings[n_context][n_features];

  float attn_input_transform[n_layers][n_features];
  float attn_input_translate[n_layers][n_features];

  float attn_query_transform[n_layers][n_heads][n_attn_features][n_features];
  float attn_query_translate[n_layers][n_heads][n_attn_features];
  float attn_key_transform[n_layers][n_heads][n_attn_features][n_features];
  float attn_key_translate[n_layers][n_heads][n_attn_features];
  float attn_value_transform[n_layers][n_heads][n_attn_features][n_features];
  float attn_value_translate[n_layers][n_heads][n_attn_features];

  float attn_output_transform[n_layers][n_features][n_heads][n_attn_features];
  float attn_output_translate[n_layers][n_features];

  float neural_input_transform[n_layers][n_features];
  float neural_input_translate[n_layers][n_features];

  float neural_activations_transform[n_layers][n_neurons][n_features];
  float neural_activations_translate[n_layers][n_neurons];

  float neural_output_transform[n_layers][n_features][n_neurons];
  float neural_output_translate[n_layers][n_features];

  float finish_transform[n_features];
  float finish_translate[n_features];
};

static size_t max_z(size_t const a, size_t const b) {
  return a > b ? a : b;
}

static void normalize(const size_t n_rows,
                      float output[static n_rows][n_features],
                      const float input[static n_rows][n_features],
                      const float transform[static n_features],
                      const float translate[static n_features]) {
  for (size_t row_i = 0; row_i < n_rows; ++row_i) {
    const float *row_input = input[row_i];
    float *row_output = output[row_i];

    float mean = 0.0f;
    for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
      mean += row_input[feature_i];
    }
    mean /= n_features;

    float variance = 0.0f;
    for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
      row_output[feature_i] = row_input[feature_i] - mean;
      variance += row_output[feature_i] * row_output[feature_i];
    }
    variance /= n_features;

    float scale = 1.0f / sqrtf(variance);
    for (int feature_i = 0; feature_i < n_features; ++feature_i) {
      row_output[feature_i] *= transform[feature_i] * scale;
      row_output[feature_i] += translate[feature_i];
    }
  }
}

int gpt2(size_t n_batches,
         size_t n_tokens,
         size_t n_past,

         float output_probs[static n_batches][n_tokens][n_vocab],

         const struct model_parameters *const p,
         batch_context_t batches[static n_batches],

         token_t const tokens[static n_batches][n_tokens]) {
  float
    (*hidden_state)[n_batches][n_tokens][n_features],

    (*attn_input)[n_tokens][n_features],
    (*attn_query)[n_heads][n_tokens][n_attn_features],
    (*attn_matrix)[n_past + n_tokens],
    (*attn_features)[n_heads][n_tokens][n_attn_features],

    (*neural_input)[n_features],
    (*neural_activations)[n_neurons];

  char
    (*tmp1)[max_z(sizeof *attn_input, max_z(sizeof *attn_matrix, sizeof *neural_activations))],
    (*tmp2)[max_z(sizeof *attn_query, max_z(sizeof *attn_features, sizeof *neural_input))];
  if ((hidden_state = malloc(sizeof *hidden_state)) == NULL ||
      (tmp1         = malloc(sizeof *tmp1)) == NULL ||
      (tmp2         = malloc(sizeof *tmp2)) == NULL)
    return -1;

  assert(sizeof *attn_input         <= sizeof *tmp1);
  attn_input         = (void *) tmp1;
  assert(sizeof *attn_query         <= sizeof *tmp2);
  attn_query         = (void *) tmp2;
  assert(sizeof *attn_matrix        <= sizeof *tmp1);
  attn_matrix        = (void *) tmp1;
  assert(sizeof *attn_features      <= sizeof *tmp2);
  attn_features      = (void *) tmp2;
  assert(sizeof *neural_input       <= sizeof *tmp2);
  neural_input       = (void *) tmp2;
  assert(sizeof *neural_activations <= sizeof *tmp1);
  neural_activations = (void *) tmp1;

  for (size_t token_i = 0; token_i < n_tokens; ++token_i) {
    size_t abs_token_pos = n_past + token_i;

    for (size_t batch_i = 0; batch_i < n_batches; ++batch_i) {
      token_t token = tokens[batch_i][token_i];

      for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
        (*hidden_state)[batch_i][token_i][feature_i]
          = p->token_embeddings[token][feature_i]
          + p->position_embeddings[abs_token_pos][feature_i];
      }
    }
  }

  for (size_t layer_i = 0; layer_i < n_layers; ++layer_i) {
    for (size_t batch_i = 0; batch_i < n_batches; ++batch_i) {
      normalize(n_tokens, *attn_input, (*hidden_state)[batch_i],
                p->attn_input_transform[layer_i], p->attn_input_translate[layer_i]);

      for (size_t head_i = 0; head_i < n_heads; ++head_i) {
        /* Three convolutions: query, key, value */
        assert((void *) attn_query != (void *) attn_input);
        for (size_t token_i = 0; token_i < n_tokens; ++token_i) {
          for (size_t attn_feature_i = 0; attn_feature_i < n_attn_features; ++attn_feature_i) {
            float *const query_output = &(*attn_query)[head_i][token_i][attn_feature_i],
              *const key_output = &batches[batch_i][layer_i].key[head_i][n_past + token_i][attn_feature_i],
              *const value_output = &batches[batch_i][layer_i].value[head_i][n_past + token_i][attn_feature_i];
            *query_output = p->attn_query_translate[layer_i][head_i][attn_feature_i];
            *key_output = p->attn_key_translate[layer_i][head_i][attn_feature_i];
            *value_output = p->attn_value_translate[layer_i][head_i][attn_feature_i];

            for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
              float const src = (*attn_input)[token_i][feature_i];
              *query_output += src * p->attn_query_transform[layer_i][head_i][attn_feature_i][feature_i];
              *key_output += src * p->attn_key_transform[layer_i][head_i][attn_feature_i][feature_i];
              *value_output += src * p->attn_value_transform[layer_i][head_i][attn_feature_i][feature_i];
            }
          }
        }
      }

      for (size_t head_i = 0; head_i < n_heads; ++head_i) {
        /* Compute attention matrix: how much attention each token should pay to past tokens */
        assert((void *) attn_matrix != (void *) attn_query);
        for (size_t dst_token_i = 0; dst_token_i < n_tokens; ++dst_token_i) {
          float max = -INFINITY;

          for (size_t src_token_i = 0; src_token_i <= n_past + dst_token_i; ++src_token_i) {
            float *const output = &(*attn_matrix)[src_token_i];
            *output = 0.0f;
            for (size_t attn_feature_i = 0; attn_feature_i < n_attn_features; ++attn_feature_i) {
              *output += (*attn_query)[head_i][dst_token_i][attn_feature_i]
                * batches[batch_i][layer_i].key[head_i][src_token_i][attn_feature_i];
            }
            *output /= sqrtf((float) n_attn_features);
            if (*output > max) {
              max = *output;
            }
          }

          float sum = 0.0f;
          for (size_t src_token_i = 0; src_token_i <= n_past + dst_token_i; ++src_token_i) {
            float *const output = &(*attn_matrix)[src_token_i];
            sum += (*output = expf(*output - max));
          }
          for (size_t src_token_i = 0; src_token_i <= n_past + dst_token_i; ++src_token_i) {
            (*attn_matrix)[src_token_i] /= sum;
          }

          /* Compute the attention value features */
          assert((void *) attn_features != (void *) attn_matrix);
          for (size_t attn_feature_i = 0; attn_feature_i < n_attn_features; ++attn_feature_i) {
            float *output = &(*attn_features)[head_i][dst_token_i][attn_feature_i];
            *output = 0.0f;
            for (size_t src_token_i = 0; src_token_i <= n_past + dst_token_i; ++src_token_i) {
              *output += (*attn_matrix)[src_token_i]
                * batches[batch_i][layer_i].value[head_i][src_token_i][attn_feature_i];
            }
          }
        }
      }

      /* Convolute and feed forward */
      for (size_t token_i = 0; token_i < n_tokens; ++token_i) {
        for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
          float *output = &(*hidden_state)[batch_i][token_i][feature_i];
          for (size_t head_i = 0; head_i < n_heads; ++head_i) {
            for (size_t attn_feature_i = 0; attn_feature_i < n_attn_features; ++attn_feature_i) {
              *output += (*attn_features)[head_i][token_i][attn_feature_i]
                * p->attn_output_transform[layer_i][feature_i][head_i][attn_feature_i];
            }
          }
          *output += p->attn_output_translate[layer_i][feature_i];
        }
      }

      for (size_t token_i = 0; token_i < n_tokens; ++token_i) {
        normalize(1, neural_input, &(*hidden_state)[batch_i][token_i],
                  p->neural_input_transform[layer_i], p->neural_input_translate[layer_i]);

        /* Convolute hidden state into neurons and perform activation */
        for (size_t neuron_i = 0; neuron_i < n_neurons; ++neuron_i) {
          float *output = &(*neural_activations)[neuron_i];
          *output = p->neural_activations_translate[layer_i][neuron_i];
          for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
            *output += (*neural_input)[feature_i]
              * p->neural_activations_transform[layer_i][neuron_i][feature_i];
          }
          float x = *output;
          *output = 0.5f * x * (1 + tanhf(M_2_SQRTPI * M_SQRT1_2 * (x + 0.044715 * x * x * x)));
        }

        /* Convolute back into hidden state */
        for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
          float *output = &(*hidden_state)[batch_i][token_i][feature_i];
          for (size_t neuron_i = 0; neuron_i < n_neurons; ++neuron_i) {
            *output += (*neural_activations)[neuron_i]
              * p->neural_output_transform[layer_i][feature_i][neuron_i];
          }
          *output += p->neural_output_translate[layer_i][feature_i];
        }
      }
    }
  }

  for (size_t batch_i = 0; batch_i < n_batches; ++batch_i) {
    normalize(n_tokens, (*hidden_state)[batch_i], (*hidden_state)[batch_i],
              p->finish_transform, p->finish_translate);

    for (size_t token_i = 0; token_i < n_tokens; ++token_i) {
      float max = -INFINITY;
      for (size_t vocab_i = 0; vocab_i < n_vocab; ++vocab_i) {
        float *output = &output_probs[batch_i][token_i][vocab_i];
        *output = 0.0f;
        for (size_t feature_i = 0; feature_i < n_features; ++feature_i) {
          *output += (*hidden_state)[batch_i][token_i][feature_i] * p->token_embeddings[vocab_i][feature_i];
        }
        if (*output > max) {
          max = *output;
        }
      }

      float sum = 0.0f;
      for (size_t vocab_i = 0; vocab_i < n_vocab; ++vocab_i) {
        float *output = &output_probs[batch_i][token_i][vocab_i];
        sum += (*output = expf(*output - max));
      }
      for (size_t vocab_i = 0; vocab_i < n_vocab; ++vocab_i) {
        output_probs[batch_i][token_i][vocab_i] /= sum;
      }
    }
  }

  free(hidden_state);
  free(tmp1);
  free(tmp2);

  return 0;
}

struct model_parameters const *load_model(const char *model_path) {
  int model_fd, save_errno;
  const struct model_parameters *model;

  if ((model_fd = open(model_path, O_RDONLY | O_CLOEXEC)) < 0)
    return NULL;
  model = mmap(NULL, sizeof *model, PROT_READ, MAP_PRIVATE, model_fd, 0);
  if (model == NULL)
    save_errno = errno;
  close(model_fd);
  if (model == NULL)
    errno = save_errno;
  return model;
}

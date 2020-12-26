#ifndef GPT2TOOLS_GPT2_H__
#define GPT2TOOLS_GPT2_H__

#include <stddef.h>
#include "hyperparameters.h"

typedef unsigned int token_t;

typedef struct model_parameters model_parameters_t;

typedef float attention_matrix_t[n_heads][n_context][n_attn_features];
typedef struct {
  attention_matrix_t key, value;
} batch_context_t[n_layers];

int gpt2(size_t n_batches,
         size_t n_tokens,
         size_t n_past,

         float output_probs[static restrict n_batches][n_tokens][n_vocab],

         const model_parameters_t *params,
         batch_context_t batches[static restrict n_batches],

         const token_t tokens[static restrict n_batches][n_tokens]);

const model_parameters_t *load_model(const char *model_path);

#endif

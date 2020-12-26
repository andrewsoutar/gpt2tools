#include <stdio.h>
#include <stdlib.h>
#include <err.h>

#include "gpt2.h"

int main(int argc, char *argv[]) {
  const model_parameters_t *params;
  float (*output_probs)[1][n_vocab];
  batch_context_t (*batches)[1];

  if ((params = load_model("model.bin")) == NULL)
    err(EXIT_FAILURE, "load_model");

  output_probs = malloc(sizeof *output_probs);
  batches = malloc(sizeof *batches);
  if (output_probs == NULL || batches == NULL)
    err(EXIT_FAILURE, "malloc");

  token_t tokens[1][1] = { 50256 };
  for (size_t i = 0; i < 50; ++i) {
    if (gpt2(sizeof *batches / sizeof **batches,
             sizeof tokens / sizeof *tokens,
             i,
             output_probs, params, *batches, tokens) != 0)
      err(EXIT_FAILURE, "gpt2");
    for (size_t vocab_i = 0; vocab_i < n_vocab; ++vocab_i) {
      if ((*output_probs)[0][vocab_i] > (*output_probs)[0][tokens[0][0]]) {
        tokens[0][0] = vocab_i;
      }
    }
    printf("%u\n", tokens[0][0]);
  }

  free(batches);
  free(output_probs);
}

/*
 * infer_export.c — Convert training weights → folded inference weights
 *
 * Usage:
 *   ./infer_export weights_v2.bin model.inf
 *
 * Reads the training weight file (MLP v2 format with BN running stats),
 * folds BatchNorm into the FC layers, and writes a compact inference
 * weight file (~800 KB) that infer.c can load.
 */

#include <stdio.h>
#include <stdlib.h>
#include "infer.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <train_weights.bin> <output.inf>\n", argv[0]);
        return 1;
    }
    const char *in_path  = argv[1];
    const char *out_path = argv[2];

    /* InferNet is ~800 KB — put on heap */
    InferNet *net = (InferNet *)malloc(sizeof(InferNet));
    if (!net) { fprintf(stderr, "Out of memory\n"); return 1; }

    printf("Reading training weights from:  %s\n", in_path);
    if (!infer_fold_from_training(net, in_path)) {
        free(net);
        return 1;
    }

    printf("Writing inference weights to:   %s\n", out_path);
    if (!infer_save(net, out_path)) {
        free(net);
        return 1;
    }

    free(net);
    printf("Done. Deploy '%s' to your edge device.\n", out_path);
    return 0;
}

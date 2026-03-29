/*
 * single.c — Run inference on a single image
 *
 * Accepts two input formats:
 *   1. PGM greyscale image (P5 binary, 28x28)
 *   2. Raw 784-byte pixel file (uint8, row-major)
 *
 * Usage:
 *   ./infer_single model.inf image.pgm
 *   ./infer_single model.inf image.raw
 *
 * Output:
 *   PRED=7  conf=99.83%
 *   probs: 0:0.00  1:0.00  ...  7:99.83  ...
 *   (followed by ASCII art preview)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "infer.h"

/* ---------------------------------------------------------------
   PGM P5 loader — only handles 28x28, maxval 255
   --------------------------------------------------------------- */
static int load_pgm(const char *path, uint8_t *pixels) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }

    char magic[3] = {0};
    int  w = 0, h = 0, maxval = 0;

    /* Header format: "P5\n<w> <h>\n<maxval>\n" — comments not handled */
    if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4) {
        fprintf(stderr, "%s: could not parse PGM header\n", path);
        fclose(f); return 0;
    }
    if (magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "%s: not a binary PGM (P5) file\n", path);
        fclose(f); return 0;
    }
    if (w != 28 || h != 28) {
        fprintf(stderr, "%s: expected 28x28, got %dx%d\n", path, w, h);
        fclose(f); return 0;
    }
    if (maxval != 255) {
        fprintf(stderr, "%s: expected maxval 255, got %d\n", path, maxval);
        fclose(f); return 0;
    }

    fgetc(f); /* skip the single whitespace byte that follows the header */

    if (fread(pixels, 1, INF_IN, f) != INF_IN) {
        fprintf(stderr, "%s: short read — file may be truncated\n", path);
        fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

/* ---------------------------------------------------------------
   Raw 784-byte loader
   --------------------------------------------------------------- */
static int load_raw(const char *path, uint8_t *pixels) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }

    size_t n = fread(pixels, 1, INF_IN, f);
    int eof  = feof(f);
    fclose(f);

    if ((int)n != INF_IN) {
        fprintf(stderr, "%s: expected %d bytes, got %zu%s\n",
                path, INF_IN, n, eof ? " (EOF)" : "");
        return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------
   Main
   --------------------------------------------------------------- */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model.inf> <image.pgm|image.raw>\n",
                argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *img_path   = argv[2];

    /* Load model */
    InferNet *net = (InferNet *)malloc(sizeof(InferNet));
    if (!net) { fprintf(stderr, "Out of memory\n"); return 1; }
    if (!infer_load(net, model_path)) { free(net); return 1; }

    /* Load image pixels */
    uint8_t pixels[INF_IN];
    int ok;
    size_t plen = strlen(img_path);
    if (plen >= 4 && strcmp(img_path + plen - 4, ".pgm") == 0)
        ok = load_pgm(img_path, pixels);
    else
        ok = load_raw(img_path, pixels);

    if (!ok) { free(net); return 1; }

    /* Single forward pass: u8 pixels → normalize → forward → softmax
       infer_forward_probs_u8 does all of this in one call and fills
       scratch.out with probabilities [0,1] summing to 1. */
    InferScratch scratch;

    /* Normalize to float once, then run forward + softmax together */
    float x[INF_IN];
    for (int i = 0; i < INF_IN; i++)
        x[i] = pixels[i] * (1.0f / 255.0f);

    int pred = infer_forward_probs(net, &scratch, x);
    /* scratch.out now holds softmax probabilities for all 10 classes */

    /* Print result */
    printf("PRED=%d  conf=%.2f%%\n", pred, scratch.out[pred] * 100.0f);
    printf("probs:");
    for (int i = 0; i < INF_OUT; i++)
        printf("  %d:%.2f%%", i, scratch.out[i] * 100.0f);
    printf("\n");

    /* ASCII art preview */
    printf("\nInput (28x28):\n");
    static const char shade[] = " .:-=+*#@";
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            int v = pixels[r * 28 + c] * 9 / 256;
            putchar(shade[v]);
            putchar(' ');
        }
        putchar('\n');
    }

    free(net);
    return 0;
}
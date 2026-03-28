/*
 * infer_single.c — Run inference on a single image
 *
 * Accepts two input formats:
 *   1. PGM greyscale image (P5 binary, 28×28)
 *   2. Raw 784-byte pixel file (uint8, row-major)
 *
 * Usage:
 *   ./infer_single model.inf image.pgm
 *   ./infer_single model.inf image.raw
 *
 * Output (one line, easy to parse from scripts):
 *   PRED=7  conf=99.83%  [0:0.00 1:0.00 2:0.01 ... 7:99.83 ...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "infer.h"

/* ---------------------------------------------------------------
   PGM P5 reader (minimal — only handles 28×28, maxval 255)
   --------------------------------------------------------------- */
static int load_pgm(const char *path, uint8_t *pixels, int expected_px) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }

    char magic[3];
    int  w, h, maxval;
    if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4 ||
        magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "%s: not a P5 PGM file\n", path);
        fclose(f); return 0;
    }
    if (w * h != expected_px) {
        fprintf(stderr, "%s: expected %d pixels, got %d×%d=%d\n",
                path, expected_px, w, h, w * h);
        fclose(f); return 0;
    }
    fgetc(f);   /* skip single whitespace after header */
    if (fread(pixels, 1, (size_t)expected_px, f) != (size_t)expected_px) {
        fprintf(stderr, "%s: read error\n", path); fclose(f); return 0;
    }
    fclose(f);
    return 1;
}

/* ---------------------------------------------------------------
   Raw 784-byte reader
   --------------------------------------------------------------- */
static int load_raw(const char *path, uint8_t *pixels, int expected_px) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    size_t n = fread(pixels, 1, (size_t)expected_px, f);
    fclose(f);
    if ((int)n != expected_px) {
        fprintf(stderr, "%s: expected %d bytes, got %zu\n",
                path, expected_px, n);
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

    InferNet *net = (InferNet *)malloc(sizeof(InferNet));
    if (!net) { fprintf(stderr, "OOM\n"); return 1; }
    if (!infer_load(net, model_path)) { free(net); return 1; }

    /* Detect format from extension */
    uint8_t pixels[INF_IN];
    int ok;
    size_t plen = strlen(img_path);
    if (plen >= 4 && strcmp(img_path + plen - 4, ".pgm") == 0)
        ok = load_pgm(img_path, pixels, INF_IN);
    else
        ok = load_raw(img_path, pixels, INF_IN);

    if (!ok) { free(net); return 1; }

    /* Run inference with probabilities */
    InferScratch scratch;
    int pred = infer_forward_probs(net, &scratch,
                   /* normalize to float */
                   (void *)NULL);

    /* Re-run properly via u8 path so probabilities are correct */
    pred = infer_forward_u8(net, &scratch, pixels);
    /* Now get probabilities too */
    {
        float x[INF_IN];
        for (int i = 0; i < INF_IN; i++) x[i] = pixels[i] * (1.0f / 255.0f);
        infer_forward_probs(net, &scratch, x);
    }

    float conf = scratch.out[pred] * 100.0f;
    printf("PRED=%d  conf=%.2f%%\n", pred, conf);
    printf("probs: ");
    for (int i = 0; i < INF_OUT; i++)
        printf("%d:%.2f%s", i, scratch.out[i] * 100.0f,
               i < INF_OUT - 1 ? "  " : "\n");

    /* ASCII art preview of the input image */
    printf("\nInput image (28×28):\n");
    const char *shade = " .:-=+*#@";
    int nshades = 9;
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            int idx = pixels[r * 28 + c] * nshades / 256;
            putchar(shade[idx]);
            putchar(' ');
        }
        putchar('\n');
    }

    free(net);
    return 0;
}

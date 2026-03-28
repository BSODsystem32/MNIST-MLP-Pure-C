/*
 * infer_bench.c — Evaluate accuracy and throughput on the MNIST test set
 *
 * Usage:
 *   ./infer_bench model.inf [data_dir]
 *
 * Loads the folded inference weights, runs all 10 000 test samples,
 * reports accuracy and samples-per-second.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "infer.h"

/* ---------------------------------------------------------------
   Minimal IDX reader — no dependency on training code
   --------------------------------------------------------------- */
static uint32_t read_be32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}

typedef struct { int count, rows, cols; uint8_t *images; uint8_t *labels; } TestData;

static int load_test(const char *img_path, const char *lbl_path, TestData *d) {
    FILE *fi = fopen(img_path, "rb");
    FILE *fl = fopen(lbl_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open: %s / %s\n", img_path, lbl_path);
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return 0;
    }
    if (read_be32(fi) != 0x00000803 || read_be32(fl) != 0x00000801) {
        fprintf(stderr, "Bad IDX magic\n"); fclose(fi); fclose(fl); return 0;
    }
    d->count = (int)read_be32(fi); read_be32(fl);   /* label count (ignored) */
    d->rows  = (int)read_be32(fi);
    d->cols  = (int)read_be32(fi);
    int px   = d->rows * d->cols;
    d->images = (uint8_t *)malloc(d->count * px);
    d->labels = (uint8_t *)malloc(d->count);
    if (!d->images || !d->labels) { fprintf(stderr, "OOM\n"); return 0; }
    if (fread(d->images, 1, (size_t)(d->count * px),  fi) != (size_t)(d->count * px) ||
        fread(d->labels, 1, (size_t)d->count,          fl) != (size_t)d->count) {
        fprintf(stderr, "Read error\n"); fclose(fi); fclose(fl); return 0;
    }
    fclose(fi); fclose(fl);
    return 1;
}

/* ---------------------------------------------------------------
   Benchmark
   --------------------------------------------------------------- */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.inf> [data_dir]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *data_dir   = (argc > 2) ? argv[2] : "./data";

    /* Load model */
    InferNet *net = (InferNet *)malloc(sizeof(InferNet));
    if (!net) { fprintf(stderr, "OOM\n"); return 1; }
    printf("Loading model: %s\n", model_path);
    if (!infer_load(net, model_path)) { free(net); return 1; }

    /* Load test set */
    char img_path[256], lbl_path[256];
    snprintf(img_path, sizeof(img_path), "%s/t10k-images.idx3-ubyte", data_dir);
    snprintf(lbl_path, sizeof(lbl_path), "%s/t10k-labels.idx1-ubyte", data_dir);
    TestData d = {0};
    printf("Loading test set: %s\n", data_dir);
    if (!load_test(img_path, lbl_path, &d)) { free(net); return 1; }
    printf("Samples: %d  (%d×%d pixels each)\n\n", d.count, d.rows, d.cols);

    /* Scratch buffer — reused for every sample */
    InferScratch scratch;
    int pixels = d.rows * d.cols;
    int correct = 0;

    /* --- Accuracy pass (also warms up caches) --- */
    for (int i = 0; i < d.count; i++) {
        int pred = infer_forward_u8(net, &scratch, d.images + i * pixels);
        if (pred == d.labels[i]) correct++;
    }
    printf("Accuracy: %d / %d = %.4f%%\n\n",
           correct, d.count, 100.0 * correct / d.count);

    /* --- Throughput benchmark: 3 warm-up passes + 10 timed passes --- */
    for (int w = 0; w < 3; w++)
        for (int i = 0; i < d.count; i++)
            infer_forward_u8(net, &scratch, d.images + i * pixels);

    int BENCH_PASSES = 10;
    clock_t t0 = clock();
    for (int p = 0; p < BENCH_PASSES; p++)
        for (int i = 0; i < d.count; i++)
            infer_forward_u8(net, &scratch, d.images + i * pixels);
    clock_t t1 = clock();

    double elapsed_s = (double)(t1 - t0) / CLOCKS_PER_SEC;
    long   total_inf = (long)BENCH_PASSES * d.count;
    printf("Throughput benchmark (%d passes × %d samples):\n",
           BENCH_PASSES, d.count);
    printf("  Total time:  %.3f s\n", elapsed_s);
    printf("  Throughput:  %.0f samples/s\n", total_inf / elapsed_s);
    printf("  Latency:     %.3f ms/sample\n",
           elapsed_s / total_inf * 1000.0);

    /* Per-class accuracy */
    int class_correct[INF_OUT] = {0};
    int class_total  [INF_OUT] = {0};
    for (int i = 0; i < d.count; i++) {
        int pred = infer_forward_u8(net, &scratch, d.images + i * pixels);
        int gt   = d.labels[i];
        class_total[gt]++;
        if (pred == gt) class_correct[gt]++;
    }
    printf("\nPer-class accuracy:\n");
    for (int c = 0; c < INF_OUT; c++)
        printf("  digit %d: %4d / %4d = %.1f%%\n",
               c, class_correct[c], class_total[c],
               100.0 * class_correct[c] / class_total[c]);

    free(net);
    free(d.images);
    free(d.labels);
    return 0;
}

#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MNIST IDX files are big-endian */
static uint32_t read_be32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}

int mnist_load(const char *image_path, const char *label_path, MnistData *out) {
    FILE *fi = fopen(image_path, "rb");
    FILE *fl = fopen(label_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open MNIST files: %s / %s\n", image_path, label_path);
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return 0;
    }

    /* Image header: magic=0x00000803, count, rows, cols */
    uint32_t magic = read_be32(fi);
    if (magic != 0x00000803) {
        fprintf(stderr, "Bad image magic: 0x%08X\n", magic);
        fclose(fi); fclose(fl);
        return 0;
    }
    out->count = (int)read_be32(fi);
    out->rows  = (int)read_be32(fi);
    out->cols  = (int)read_be32(fi);

    /* Label header: magic=0x00000801, count */
    uint32_t lmagic = read_be32(fl);
    if (lmagic != 0x00000801) {
        fprintf(stderr, "Bad label magic: 0x%08X\n", lmagic);
        fclose(fi); fclose(fl);
        return 0;
    }
    int lcount = (int)read_be32(fl);
    if (lcount != out->count) {
        fprintf(stderr, "Image/label count mismatch: %d vs %d\n", out->count, lcount);
        fclose(fi); fclose(fl);
        return 0;
    }

    int pixels = out->rows * out->cols;
    out->images = (float *)malloc(out->count * pixels * sizeof(float));
    out->labels = (uint8_t *)malloc(out->count * sizeof(uint8_t));
    if (!out->images || !out->labels) {
        fprintf(stderr, "Out of memory\n");
        free(out->images); free(out->labels);
        fclose(fi); fclose(fl);
        return 0;
    }

    /* Read raw bytes, normalize to [0, 1] */
    uint8_t *raw = (uint8_t *)malloc(pixels);
    for (int i = 0; i < out->count; i++) {
        if (fread(raw, 1, (size_t)pixels, fi) != (size_t)pixels) { free(raw); fclose(fi); fclose(fl); free(out->images); free(out->labels); return 0; }
        for (int p = 0; p < pixels; p++)
            out->images[i * pixels + p] = raw[p] / 255.0f;
    }
    free(raw);

    if (fread(out->labels, 1, (size_t)out->count, fl) != (size_t)out->count) { fclose(fi); fclose(fl); free(out->images); free(out->labels); return 0; }

    fclose(fi);
    fclose(fl);
    return 1;
}

void mnist_free(MnistData *d) {
    free(d->images);
    free(d->labels);
    d->images = NULL;
    d->labels = NULL;
    d->count  = 0;
}

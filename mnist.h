#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

typedef struct {
    int count;
    int rows;
    int cols;
    float *images;  /* [count × rows × cols], normalized to [0,1] */
    uint8_t *labels;
} MnistData;

/* Load images and labels from IDX binary files.
   Returns 1 on success, 0 on failure. */
int mnist_load(const char *image_path, const char *label_path, MnistData *out);
void mnist_free(MnistData *d);

#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mnist.h"
#include "network.h"
#include "augment.h"

/* ---------------------------------------------------------------
   Hyperparameters
   --------------------------------------------------------------- */
#define EPOCHS       30
#define BATCH_SIZE   128
#define LR_INIT      0.05f
#define LR_DECAY     0.92f
#define DROPOUT_RATE 0.3f

static void shuffle(int *idx, int n) {
    for (int i = n-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

static float evaluate(Network *net, const MnistData *d) {
    int pixels  = d->rows * d->cols;
    int correct = 0;
    float *batch_x = (float *)malloc(BATCH_SIZE * pixels * sizeof(float));
    for (int start = 0; start < d->count; start += BATCH_SIZE) {
        int bs = BATCH_SIZE;
        if (start + bs > d->count) bs = d->count - start;
        memcpy(batch_x, d->images + start*pixels, bs*pixels*sizeof(float));
        net_forward(net, batch_x, bs, 0, 0.0f);
        correct += net_correct(net, d->labels + start, bs);
    }
    free(batch_x);
    return 100.0f * correct / d->count;
}

/* ---------------------------------------------------------------
   Per-thread augmentation scratch — allocated once, reused every batch
   --------------------------------------------------------------- */
typedef struct {
    float dst[AUG_PX];
    float tmp[AUG_PX];
    RNG   rng;
} ThreadScratch;

static void train(Network *net, const MnistData *train_d,
                  const MnistData *test_d, const AugmentCfg *aug) {
    int pixels = train_d->rows * train_d->cols;
    int n      = train_d->count;

    int     *idx     = (int     *)malloc(n * sizeof(int));
    float   *batch_x = (float   *)malloc(BATCH_SIZE * pixels * sizeof(float));
    uint8_t *batch_y = (uint8_t *)malloc(BATCH_SIZE * sizeof(uint8_t));

    /* One scratch struct per thread — avoids false sharing and per-iter malloc */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    ThreadScratch *scratch = (ThreadScratch *)malloc(nthreads * sizeof(ThreadScratch));

    /* Seed each thread's RNG differently */
    unsigned base = (unsigned)time(NULL);
    for (int t = 0; t < nthreads; t++)
        rng_seed(&scratch[t].rng, base ^ (uint32_t)(t * 2654435761u));

    for (int i = 0; i < n; i++) idx[i] = i;

    float lr = LR_INIT;
    printf("%-6s %-8s %-10s %-10s %-10s\n",
           "Epoch", "LR", "Loss", "Train%", "Test%");
    printf("----------------------------------------------\n");

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        shuffle(idx, n);

        float total_loss    = 0.0f;
        int   total_correct = 0;
        int   steps         = 0;

        for (int start = 0; start + BATCH_SIZE <= n; start += BATCH_SIZE) {

            /* ---- Augment batch in parallel ---- */
            #ifdef _OPENMP

            #pragma omp parallel for schedule(static)

            #endif
            for (int b = 0; b < BATCH_SIZE; b++) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                ThreadScratch *ts = &scratch[tid];
                const float *src  = train_d->images + idx[start+b] * pixels;

                augment_apply(ts->dst, src, ts->tmp, &ts->rng, aug);
                memcpy(batch_x + b * pixels, ts->dst, pixels * sizeof(float));
                batch_y[b] = train_d->labels[idx[start+b]];
            }

            /* ---- Forward / backward (matrix ops use OMP internally) ---- */
            net_forward(net, batch_x, BATCH_SIZE, 1, DROPOUT_RATE);
            total_loss    += net_loss(net, batch_y, BATCH_SIZE);
            total_correct += net_correct(net, batch_y, BATCH_SIZE);
            net_backward(net, batch_x, batch_y, BATCH_SIZE, lr, DROPOUT_RATE);
            steps++;
        }

        float train_acc = 100.0f * total_correct / (steps * BATCH_SIZE);
        float test_acc  = evaluate(net, test_d);
        printf("%-6d %-8.5f %-10.4f %-10.2f %-10.2f\n",
               epoch, lr, total_loss/steps, train_acc, test_acc);
        lr *= LR_DECAY;
    }

    free(idx); free(batch_x); free(batch_y); free(scratch);
}

int main(int argc, char **argv) {
    const char *data_dir    = (argc > 1) ? argv[1] : "./data";
    const char *weights_out = (argc > 2) ? argv[2] : "weights_v2.bin";

    srand((unsigned)time(NULL));

    char train_img[256], train_lbl[256], test_img[256], test_lbl[256];
    snprintf(train_img, sizeof(train_img), "%s/train-images-idx3-ubyte", data_dir);
    snprintf(train_lbl, sizeof(train_lbl), "%s/train-labels-idx1-ubyte", data_dir);
    snprintf(test_img,  sizeof(test_img),  "%s/t10k-images-idx3-ubyte",  data_dir);
    snprintf(test_lbl,  sizeof(test_lbl),  "%s/t10k-labels-idx1-ubyte",  data_dir);

    printf("Loading MNIST from %s ...\n", data_dir);
    MnistData train_d = {0}, test_d = {0};
    if (!mnist_load(train_img, train_lbl, &train_d)) return 1;
    if (!mnist_load(test_img,  test_lbl,  &test_d))  return 1;
    printf("Train: %d   Test: %d\n\n", train_d.count, test_d.count);

    AugmentCfg aug = augment_default();
    aug.elastic_alpha = 8.0f;
    aug.elastic_sigma = 3.0f;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    printf("Threads: %d\n", nthreads);
    printf("Config:  epochs=%d  batch=%d  lr=%.3f->%.2f  dropout=%.1f\n",
           EPOCHS, BATCH_SIZE, LR_INIT, LR_DECAY, DROPOUT_RATE);
    printf("Augment: shift=+/-%dpx  rotate=+/-%.0fdeg  scale=+/-%.0f%%"
           "  elastic(alpha=%.1f sigma=%.1f)\n\n",
           aug.max_shift, aug.max_angle_deg, aug.scale_delta*100.0f,
           aug.elastic_alpha, aug.elastic_sigma);

    Network *net = net_create();
    if (!net) { fprintf(stderr, "Out of memory\n"); return 1; }

    train(net, &train_d, &test_d, &aug);

    printf("----------------------------------------------\n");
    printf("Final test accuracy: %.2f%%\n", evaluate(net, &test_d));
    net_save(net, weights_out);
    net_free(net);
    mnist_free(&train_d);
    mnist_free(&test_d);
    return 0;
}

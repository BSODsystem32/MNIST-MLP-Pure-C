#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "mnist.h"
#include "network.h"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
    #include <sys/time.h>
#endif

/* ---------------------------------------------------------------
   Timer for Windows & Linux
   --------------------------------------------------------------- */
double get_time_ms() {
#ifdef _WIN32
    // Windows
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / freq.QuadPart;
#else
    // Linux/POSIX
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

/* ---------------------------------------------------------------
   Hyperparameters
   --------------------------------------------------------------- */
#define EPOCHS       25
#define BATCH_SIZE   128
#define LR_INIT      0.05f
#define LR_DECAY     0.92f
#define DROPOUT_RATE 0.3f    /* fraction of neurons to drop */

/* ---------------------------------------------------------------
   Fisher-Yates shuffle
   --------------------------------------------------------------- */
static void shuffle(int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

/* ---------------------------------------------------------------
   Evaluate accuracy (inference mode: no dropout, running BN stats)
   --------------------------------------------------------------- */
static float evaluate(Network *net, const MnistData *d) {
    int pixels  = d->rows * d->cols;
    int correct = 0;
    float *batch_x = (float *)malloc(BATCH_SIZE * pixels * sizeof(float));

    for (int start = 0; start < d->count; start += BATCH_SIZE) {
        int bs = BATCH_SIZE;
        if (start + bs > d->count) bs = d->count - start;
        memcpy(batch_x, d->images + start * pixels,
               bs * pixels * sizeof(float));
        /* training=0, dropout_rate ignored */
        net_forward(net, batch_x, bs, 0, 0.0f);
        correct += net_correct(net, d->labels + start, bs);
    }
    free(batch_x);
    return 100.0f * correct / d->count;
}

/* ---------------------------------------------------------------
   Training loop
   --------------------------------------------------------------- */
static void train(Network *net, const MnistData *train_d,
                  const MnistData *test_d) {
    int pixels = train_d->rows * train_d->cols;
    int n      = train_d->count;

    int     *idx     = (int     *)malloc(n * sizeof(int));
    float   *batch_x = (float   *)malloc(BATCH_SIZE * pixels * sizeof(float));
    uint8_t *batch_y = (uint8_t *)malloc(BATCH_SIZE * sizeof(uint8_t));

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
            for (int b = 0; b < BATCH_SIZE; b++) {
                int s = idx[start + b];
                memcpy(batch_x + b * pixels,
                       train_d->images + s * pixels,
                       pixels * sizeof(float));
                batch_y[b] = train_d->labels[s];
            }
            /* Forward in training mode */
            net_forward(net, batch_x, BATCH_SIZE, 1, DROPOUT_RATE);
            total_loss    += net_loss(net, batch_y, BATCH_SIZE);
            total_correct += net_correct(net, batch_y, BATCH_SIZE);
            net_backward(net, batch_x, batch_y, BATCH_SIZE, lr, DROPOUT_RATE);
            steps++;
        }

        float train_acc = 100.0f * total_correct / (steps * BATCH_SIZE);
        float test_acc  = evaluate(net, test_d);

        printf("%-6d %-8.5f %-10.4f %-10.2f %-10.2f\n",
               epoch, lr, total_loss / steps, train_acc, test_acc);

        lr *= LR_DECAY;
    }

    free(idx);
    free(batch_x);
    free(batch_y);
}

/* ---------------------------------------------------------------
   Entry point
   --------------------------------------------------------------- */
int main(int argc, char **argv) {
    const char *data_dir    = (argc > 1) ? argv[1] : "./data";
    const char *weights_out = (argc > 2) ? argv[2] : "weights_v2.bin";

    const double start = get_time_ms();
    srand((unsigned)time(NULL));

    char train_img[256], train_lbl[256], test_img[256], test_lbl[256];
    snprintf(train_img, sizeof(train_img), "%s/train-images.idx3-ubyte", data_dir);
    snprintf(train_lbl, sizeof(train_lbl), "%s/train-labels.idx1-ubyte", data_dir);
    snprintf(test_img,  sizeof(test_img),  "%s/t10k-images.idx3-ubyte",  data_dir);
    snprintf(test_lbl,  sizeof(test_lbl),  "%s/t10k-labels.idx1-ubyte",  data_dir);

    printf("Loading MNIST from %s ...\n", data_dir);
    MnistData train_d = {0}, test_d = {0};
    if (!mnist_load(train_img, train_lbl, &train_d)) return 1;
    if (!mnist_load(test_img,  test_lbl,  &test_d))  return 1;
    printf("Train: %d   Test: %d\n\n", train_d.count, test_d.count);

    printf("Config: epochs=%d  batch=%d  lr=%.3f decay=%.2f  dropout=%.1f\n",
           EPOCHS, BATCH_SIZE, LR_INIT, LR_DECAY, DROPOUT_RATE);
    printf("        momentum=%.2f  BN_eps=%.0e  BN_momentum=%.2f\n\n",
           MOMENTUM, (double)BN_EPS, (double)BN_MOMENTUM);

    Network *net = net_create();
    if (!net) { fprintf(stderr, "Out of memory\n"); return 1; }

    train(net, &train_d, &test_d);

    printf("----------------------------------------------\n");
    printf("Final test accuracy: %.2f%%\n", evaluate(net, &test_d));

    net_save(net, weights_out);
    net_free(net);
    mnist_free(&train_d);
    mnist_free(&test_d);

    const double end = get_time_ms();
    printf("Time: %lf ms\n", end - start);

    return 0;
}

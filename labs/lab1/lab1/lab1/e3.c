#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>

// Number of threads to create
#define NUM_THREADS 4

void *runWorkerThreadsStd(void *work);
void mult_std(float* a, float* b, float* r, int num);
void mult_vect(float* a, float* b, float* r, int num);

typedef struct work {
    float *a;
    float *b;
    float *r;
    int num;
} Work;

void *runWorkerThreadsStd(void *work) {
    Work* w = (Work *) work;
    mult_std(w->a, w->b, w->r, w->num);
    pthread_exit(NULL);
}

void mult_std(float* a, float* b, float* r, int num)
{
    for (int i = 0; i < num; i++)
    {
        r[i] = a[i] * b[i];
    }
}

void mult_vect(float* a, float* b, float* r, int num)
{
    float32x4_t va, vb, vr;
    for (int i = 0; i < num; i +=4)
    {
        va = vld1q_f32(&a[i]);
        vb = vld1q_f32(&b[i]);
        vr = vmulq_f32(va, vb);
        vst1q_f32(&r[i], vr);
    }
}

int main(int argc, char *argv[]) {
    int num = 100000000;
    float *a = (float*)aligned_alloc(16, num*sizeof(float));
    float *b = (float*)aligned_alloc(16, num*sizeof(float));
    float *r = (float*)aligned_alloc(16, num*sizeof(float));

    for (int i = 0; i < num; i++)
    {
        a[i] = (i % 127)*0.1457f;
        b[i] = (i % 331)*0.1231f;
    }

    struct timespec ts_start;
    struct timespec ts_end_1;
    struct timespec ts_end_2;

    pthread_t threads[NUM_THREADS];
    Work splitUpData[NUM_THREADS];
    int rc;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for(int i=0; i < NUM_THREADS; i++) {
        splitUpData[i].a = a + (num/NUM_THREADS * i);
        splitUpData[i].b = b + (num/NUM_THREADS * i);
        splitUpData[i].r = r + (num/NUM_THREADS * i);
        splitUpData[i].num = (num/NUM_THREADS);
    }

    for(long t = 0; t < NUM_THREADS; t++) {
        printf("Thread #%ld is activated\n\n", t);
        rc = pthread_create(&threads[t], NULL, runWorkerThreadsStd, &splitUpData[t]);
        if(rc) {
            printf("Error: Unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    // Join the threads
    for(long t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    // mult_std(a, b, r, num);
    clock_gettime(CLOCK_MONOTONIC, &ts_end_1);
    mult_vect(a, b, r, num);
    clock_gettime(CLOCK_MONOTONIC, &ts_end_2);

    double duration_std = (ts_end_1.tv_sec - ts_start.tv_sec) +
    (ts_end_1.tv_nsec - ts_start.tv_nsec) * 1e-9;
    double duration_vec = (ts_end_2.tv_sec - ts_end_1.tv_sec) +
    (ts_end_2.tv_nsec - ts_end_1.tv_nsec) * 1e-9;

    printf("-- Elapsed time std: %f\n", duration_std);
    printf("-- Elapsed time vec: %f\n", duration_vec);
    free(a);
    free(b);
    free(r);
    return 0;
}
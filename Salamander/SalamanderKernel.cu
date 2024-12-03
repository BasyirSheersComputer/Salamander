#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <cstdlib>  // For malloc, free, and rand
#include <ctime>    // For time (used for srand)

constexpr int NUM_USERS = 1000;
constexpr int NUM_ITEMS = 1000;
constexpr int K = 10; // Number of latent features
constexpr int NUM_EPOCHS = 20;
constexpr float LEARNING_RATE = 0.01f;
constexpr float LAMBDA = 0.02f; // Regularization parameter

// CUDA kernel for matrix factorization
__global__ void matrixFactorizationSGD(
    int* userIds, int* itemIds, float* ratings,
    float* P, float* Q, int numInteractions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numInteractions) return;

    int u = userIds[idx];
    int i = itemIds[idx];
    float r = ratings[idx];

    float* Pu = &P[u * K];
    float* Qi = &Q[i * K];

    // Compute the dot product for predicted rating
    float pred = 0.0f;
    for (int k = 0; k < K; ++k) {
        pred += Pu[k] * Qi[k];
    }

    // Compute error
    float err = r - pred;

    // Update P and Q matrices
    for (int k = 0; k < K; ++k) {
        float gradPu = err * Qi[k] - LAMBDA * Pu[k];
        float gradQi = err * Pu[k] - LAMBDA * Qi[k];

        Pu[k] += LEARNING_RATE * gradPu;
        Qi[k] += LEARNING_RATE * gradQi;
    }
}

// Utility function for random data generation (host-side)
void generateData(int numUsers, int numItems, int numInteractions,
    int* userIds, int* itemIds, float* ratings) {
    srand(time(0));
    for (int i = 0; i < numInteractions; ++i) {
        userIds[i] = rand() % numUsers;
        itemIds[i] = rand() % numItems;
        ratings[i] = static_cast<float>(rand() % 5 + 1); // Ratings between 1 and 5
    }
}

// Exposed function for running the CUDA kernel
extern "C" void runMatrixFactorization(
    int* userIds, int* itemIds, float* ratings,
    float* P, float* Q,
    int numUsers, int numItems, int numInteractions, int numEpochs) {

    // Allocate device memory
    int* d_userIds, * d_itemIds;
    float* d_ratings, * d_P, * d_Q;

    cudaMalloc(&d_userIds, numInteractions * sizeof(int));
    cudaMalloc(&d_itemIds, numInteractions * sizeof(int));
    cudaMalloc(&d_ratings, numInteractions * sizeof(float));
    cudaMalloc(&d_P, numUsers * K * sizeof(float));
    cudaMalloc(&d_Q, numItems * K * sizeof(float));

    cudaMemcpy(d_userIds, userIds, numInteractions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_itemIds, itemIds, numInteractions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratings, ratings, numInteractions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, numUsers * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, numItems * K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 256;
    int gridSize = (numInteractions + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        matrixFactorizationSGD << <gridSize, blockSize >> > (d_userIds, d_itemIds, d_ratings, d_P, d_Q, numInteractions);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(P, d_P, numUsers * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, d_Q, numItems * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_userIds);
    cudaFree(d_itemIds);
    cudaFree(d_ratings);
    cudaFree(d_P);
    cudaFree(d_Q);
}

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Function declaration for the CUDA kernel
extern "C" void runMatrixFactorization(
    int* userIds, int* itemIds, float* ratings,
    float* P, float* Q,
    int numUsers, int numItems, int numInteractions, int numEpochs);

void generateData(int numUsers, int numItems, int numInteractions,
    std::vector<int>& userIds,
    std::vector<int>& itemIds,
    std::vector<float>& ratings) {
    srand(time(0));
    for (int i = 0; i < numInteractions; ++i) {
        userIds.push_back(rand() % numUsers);
        itemIds.push_back(rand() % numItems);
        ratings.push_back(static_cast<float>(rand() % 5 + 1)); // Ratings between 1 and 5
    }
}

int main() {
    int numUsers, numItems, numInteractions, numEpochs;

    // User input
    std::cout << "=== CUDA Matrix Factorization Test ===\n";
    std::cout << "Enter number of users: ";
    std::cin >> numUsers;
    std::cout << "Enter number of items: ";
    std::cin >> numItems;
    std::cout << "Enter number of interactions: ";
    std::cin >> numInteractions;
    std::cout << "Enter number of epochs: ";
    std::cin >> numEpochs;

    // Data structures
    std::vector<int> userIds, itemIds;
    std::vector<float> ratings;

    // Generate data
    generateData(numUsers, numItems, numInteractions, userIds, itemIds, ratings);

    // Allocate memory for P and Q matrices
    std::vector<float> P(numUsers * 10, 0.1f); // Initialized to 0.1
    std::vector<float> Q(numItems * 10, 0.1f); // Initialized to 0.1

    // Run the CUDA function
    runMatrixFactorization(userIds.data(), itemIds.data(), ratings.data(),
        P.data(), Q.data(),
        numUsers, numItems, numInteractions, numEpochs);

    // Display results
    std::cout << "\n=== Results (Partial View) ===\n";
    std::cout << "User Features (P):\n";
    for (int i = 0; i < std::min(numUsers, 5); ++i) {
        for (int k = 0; k < 10; ++k) {
            std::cout << P[i * 10 + k] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Item Features (Q):\n";
    for (int i = 0; i < std::min(numItems, 5); ++i) {
        for (int k = 0; k < 10; ++k) {
            std::cout << Q[i * 10 + k] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Matrix factorization completed successfully!\n";
    return 0;
}

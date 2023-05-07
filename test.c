#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 28
#define COLS 28
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 10

float tanh_fn(float x) {
    return tanh(x);
}

float softmax_fn(float x) {
    return exp(x);
}

int predict(float input[ROWS * COLS], float p1_b[HIDDEN_SIZE], float p1_w[ROWS * COLS * HIDDEN_SIZE], float p2_b[OUTPUT_SIZE], float p2_w[HIDDEN_SIZE * OUTPUT_SIZE]) {
    float l1[HIDDEN_SIZE];
    float l2[OUTPUT_SIZE];

    // Compute the result of the first layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = 0;
        for (int j = 0; j < ROWS * COLS; j++) {
            sum += input[j] * p1_w[j * HIDDEN_SIZE + i];
        }
        l1[i] = tanh_fn(sum + p1_b[i]);
    }

    // Compute the result of the second layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += l1[j] * p2_w[j * OUTPUT_SIZE + i];
        }
        l2[i] = softmax_fn(sum + p2_b[i]);
    }

    // Find the digit with the highest probability and return it
    int max_idx = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (l2[i] > l2[max_idx]) max_idx = i;
    }
    return max_idx;
}

int main() {
    // Read input image and model parameters
    FILE *input_file = fopen("./input.txt", "r");
    FILE *p1_b_file = fopen("./p_1_b.txt", "r");
    FILE *p1_w_file = fopen("./p_1_w.txt", "r");
    FILE *p2_b_file = fopen("./p_2_b.txt", "r");
    FILE *p2_w_file = fopen("./p_2_w.txt", "r");

    float input[ROWS * COLS];
    float p1_b[HIDDEN_SIZE], p1_w[ROWS * COLS * HIDDEN_SIZE], p2_b[OUTPUT_SIZE], p2_w[HIDDEN_SIZE * OUTPUT_SIZE];
    for (int i = 0; i < ROWS * COLS; i++) fscanf(input_file, "%f", &input[i]);
    for (int i = 0; i < HIDDEN_SIZE; i++) fscanf(p1_b_file, "%f", &p1_b[i]);
    for (int i = 0; i < ROWS * COLS * HIDDEN_SIZE; i++) fscanf(p1_w_file, "%f", &p1_w[i]);
    for (int i = 0; i < OUTPUT_SIZE; i++) fscanf(p2_b_file, "%f", &p2_b[i]);
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) fscanf(p2_w_file, "%f", &p2_w[i]);

    // Output predicted result
    printf("Predicted number: %d\n", predict(input, p1_b, p1_w, p2_b, p2_w));

    return 0;
}

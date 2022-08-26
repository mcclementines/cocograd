#include <stdlib.h>
#include <time.h>

#include "cocograd.h"

struct Neuron *init_neuron(int nin) {
    srand(time(NULL));

    struct Neuron *neuron = malloc(sizeof (struct ValueList));
    struct ValueList *weights = malloc(sizeof (struct ValueList));
    struct Value *bias;

    weights->list = malloc(nin * sizeof(struct Value *));
    weights->size = nin;

    for (int i = 0; i < nin; i++) {
        double rand_double = (double) rand() * (1 - - 1) / (double) RAND_MAX + -1;

        struct Value *val = init_value(rand_double);

        weights->list[i] = val;
    }

    bias = init_value((double) rand() * (1 - - 1) / (double) RAND_MAX + -1);

    neuron->weights = weights;
    neuron->bias = bias;

    return neuron;
}

struct Value *eval_neuron(struct Neuron *neuron, struct ValueList *list) {
    if (neuron->weights->size != list->size) return NULL;

    struct Value *val = init_value(0);

    for (int i = 0; i < list->size; i++) {
        val = add_values(val, mul_values(neuron->weights->list[i], list->list[i]));
    }

    val = add_values(val, neuron->bias);

    val = tanh_value(val);

    return val;
}
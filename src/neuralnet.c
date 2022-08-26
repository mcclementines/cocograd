#include <stdlib.h>
#include <time.h>

#include "cocograd.h"

void init_net() {
   srand(time(NULL));
}

struct Neuron *init_neuron(int nin) {
    struct Neuron *neuron = malloc(sizeof(struct ValueList));
    struct ValueList *weights = malloc(sizeof(struct ValueList));
    struct Value *bias;

    weights->list = malloc(nin * sizeof(struct Value *));
    weights->size = nin;

    for (int i = 0; i < nin; i++)
    {
        double rand_double = (double)rand() * (1 - -1) / (double)RAND_MAX + -1;

        struct Value *val = init_value(rand_double);

        weights->list[i] = val;
    }

    bias = init_value((double)rand() * (1 - -1) / (double)RAND_MAX + -1);

    neuron->weights = weights;
    neuron->bias = bias;

    return neuron;
}

struct Value *eval_neuron(struct Neuron *neuron, struct ValueList *xs) {
    if (neuron->weights->size != xs->size) return NULL;

    struct Value *val = init_value(0);

    for (int i = 0; i < xs->size; i++) {
        val = add_values(val, mul_values(neuron->weights->list[i], xs->list[i]));
    }

    val = add_values(val, neuron->bias);

    val = tanh_value(val);

    return val;
}

struct Layer *init_layer(int nin, int nout) {
    struct Layer *list = malloc(sizeof (struct ValueList));

    list->neurons = malloc(nout * sizeof(struct Value *));
    list->size = nout;

    for (int i = 0; i < nout; i++) {
        list->neurons[i] = init_neuron(nin);
    }

    return list;
}

struct ValueList *eval_layer(struct Layer *layer, struct ValueList *xs) {
    struct ValueList *vals = init_value_list(NULL, 0);

    for (int i = 0; i < layer->size; i++) {
        add_value_to_list(eval_neuron(layer->neurons[i], xs), vals);
    }

    return vals;
}

bool is_neuron_in_list(struct Neuron *neuron, struct Layer *list) {
        for (int i = 0; i < list->size; i++) {
        if (neuron == (list->neurons)[i])
            return true;
    }

    return false;
}

void add_neuron_to_list(struct Neuron *neuron, struct Layer *list) {
    struct Neuron **new_list = malloc(1 + list->size * sizeof(struct Neuron *));

    for (int i = 0; i < list->size; i++)
        new_list[i] = (list->neurons)[i];

    new_list[list->size] = neuron;

    free(list->neurons);
    
    list->neurons = new_list;
    list->size += 1;
}
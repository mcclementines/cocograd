#ifndef COCOGRAD_NET
#define COCOGRAD_NET

#include "grad.h"

struct Neuron {
    struct ValueList *weights;
    struct Value *bias;
};

struct Layer {
    struct Neuron **neurons;
    int size;
};

void init_net();

struct Neuron *init_neuron();

struct Value *eval_neuron(struct Neuron *neuron, struct ValueList *xs);

struct Layer *init_layer(int nin, int nout);

struct ValueList *eval_layer(struct Layer *layer, struct ValueList *xs);

bool is_neuron_in_layer(struct Neuron *neuron, struct Layer *list);

void add_neuron_to_layer(struct Neuron *neuron, struct Layer *list);

#endif
#ifndef COCOGRAD_NET
#define COCOGRAD_NET

#include "grad.h"

struct Neuron {
    struct ValueList *weights;
    struct Value *bias;
};

struct Neuron *init_neuron();

struct Value *eval_neuron(struct Neuron *neuron, struct ValueList *list);

#endif
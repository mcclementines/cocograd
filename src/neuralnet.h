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

struct MLP {
    struct Layer **layers;
    int size;
};

void init_net();

struct Neuron *init_neuron();

struct ValueList *neuron_params(struct Neuron *neuron);

struct Value *eval_neuron(struct Neuron *neuron, struct ValueList *xs);

struct Layer *init_layer(int nin, int nout);

struct ValueList *layer_params(struct Layer *layer);

struct ValueList *eval_layer(struct Layer *layer, struct ValueList *xs);

bool is_neuron_in_layer(struct Neuron *neuron, struct Layer *list);

void add_neuron_to_layer(struct Neuron *neuron, struct Layer *list);

struct MLP *init_mlp(int nin, int outs[], int nouts);

struct ValueList *mlp_params(struct MLP *mlp);

struct ValueList *eval_mlp(struct MLP *mlp, struct ValueList *xs);

struct Value *mean_squared_loss(struct Value *pred, struct Value *truth);

void train_network_with_iteration(struct MLP *mlp, struct ValueList **inputs, struct ValueList *truths, int nin, double step, int iterations);

#endif
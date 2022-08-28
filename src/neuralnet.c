#include <stdlib.h>
#include <stdio.h>
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

struct ValueList *neuron_params(struct Neuron *neuron) {
    struct ValueList *vals = init_value_list(0, 0);

    add_value_lists(vals, neuron->weights);
    add_value_to_list(neuron->bias, vals);

    return vals;
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

struct ValueList *layer_params(struct Layer *layer) {
    struct ValueList *layer_params = init_value_list(0, 0);

    for (int i = 0; i < layer->size; i++) {
        struct ValueList *params = neuron_params(layer->neurons[i]);

        add_value_lists(layer_params, params);

        free(params);
    }

    return layer_params;
}

struct ValueList *eval_layer(struct Layer *layer, struct ValueList *xs) {
    struct ValueList *vals = init_value_list(NULL, 0);

    for (int i = 0; i < layer->size; i++) {
        add_value_to_list(eval_neuron(layer->neurons[i], xs), vals);
    }

    return vals;
}

bool is_neuron_in_layer(struct Neuron *neuron, struct Layer *list) {
        for (int i = 0; i < list->size; i++) {
        if (neuron == (list->neurons)[i])
            return true;
    }

    return false;
}

void add_neuron_to_layer(struct Neuron *neuron, struct Layer *list) {
    struct Neuron **new_list = malloc(1 + list->size * sizeof(struct Neuron *));

    for (int i = 0; i < list->size; i++)
        new_list[i] = (list->neurons)[i];

    new_list[list->size] = neuron;

    free(list->neurons);
    
    list->neurons = new_list;
    list->size += 1;
}

struct MLP *init_mlp(int nin, int outs[], int nouts) {
    struct MLP *mlp = malloc(sizeof (struct MLP));

    mlp->layers = malloc((nouts+1) * sizeof(struct Value *));
    mlp->size = nouts;

    for (int i = 0; i < nouts; i++) {
        if (i == 0) {
            mlp->layers[i] = init_layer(nin, outs[i+1]);
            continue;
        }

        mlp->layers[i] = init_layer(outs[i-1], outs[i]);
    }

    return mlp;
}

struct ValueList *mlp_params(struct MLP *mlp) {
    struct ValueList *mlp_params = init_value_list(0, 0);

    for (int i = 0; i < mlp->size; i++) {
        struct ValueList *params = layer_params(mlp->layers[i]);

        add_value_lists(mlp_params, params);

        free(params);
    }

    return mlp_params;
}

struct ValueList *eval_mlp(struct MLP *mlp, struct ValueList *xs) {
    struct ValueList *list = xs;

    for (int i = 0; i < mlp->size; i++)  {
        list = eval_layer(mlp->layers[i], list);
    }

    return list;
}

struct Value *mean_squared_loss(struct Value *pred, struct Value *truth) {
    return pow_value_double(sub_values(pred, truth), 2.0);
}

void train_network() {}

void train_network_with_iteration(struct MLP *mlp, struct ValueList **inputs, struct ValueList *truths, int nin, double step, int iterations) {
    for (int iters = 0; iters < iterations; iters++) {
        struct ValueList *ypred = init_value_list(0, 0);
        
        for (int i = 0; i < nin; i++) {
            add_value_lists(ypred, eval_mlp(mlp, inputs[i]));
        }

        struct Value *loss = init_value(0);
        for (int i = 0; i < nin; i++) {
            loss = add_values(loss, mean_squared_loss(ypred->list[i], truths->list[i]));
        }

        printf("Iteration %d loss: %f\n", iters, loss->data);

        backwards(loss);

        struct ValueList *params = mlp_params(mlp);

        for (int i = 0; i < params->size; i++) {
            params->list[i]->data += -1.0 * step * params->list[i]->grad;
            params->list[i]->grad = 0.0;
        }
    }
}
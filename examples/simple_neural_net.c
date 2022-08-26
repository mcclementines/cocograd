#include "cocograd.h"

int main() {
    struct Neuron *n = init_neuron(2);

    double xs[2] = {2, 3};
    struct ValueList *x = init_value_list(xs, 2);

    struct Value *val = eval_neuron(n, x);

    print_value(val);
}
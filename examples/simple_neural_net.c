#include "cocograd.h"

int main() {
    init_net();

    struct Layer *layer = init_layer(2, 4);

    double xs[2] = {2, 3};
    struct ValueList *x = init_value_list(xs, 2);

    struct ValueList *vals = eval_layer(layer, x);

    print_value_list(vals);
}
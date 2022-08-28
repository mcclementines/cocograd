#include "cocograd.h"

#include <stdio.h>

int main() {
    init_net();

    int layers[3] = {4, 4, 1};
    struct MLP *mlp = init_mlp(3, layers, 3);

    double xs[4*3] = {2.0, 3.0, -1.0,
                      3.0, -1.0, 0.5,
                      0.5, 1.0, 1.0,
                      1.0, 1.0, 1.0};
    
    double ys[4] = {1.0, -1.0, -1.0, 1.0};

    struct ValueList **inputs = init_value_list_2d(xs, 4, 3);
    struct ValueList *truths = init_value_list(ys, 4);

    train_network_with_iteration(mlp, inputs, truths, 4, .05, 200);

    print_value_list(eval_mlp_for_inputs(mlp, inputs, 4));
}
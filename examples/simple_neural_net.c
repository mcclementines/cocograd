#include "cocograd.h"

int main() {
    init_net();

    int layers[3] = {4, 4, 1};
    struct MLP *mlp = init_mlp(3, layers, 3);

    double xs[4*3] = {2.0, 3.0, -1.0,
                      3.0, -1.0, 0.5,
                      0.5, 1.0, 1.0,
                      1.0, 1.0, 1.0};

    struct ValueList **inputs = value_list_arr(xs, 4, 3);
    struct ValueList *outs = init_value_list(0, 0);

    for (int i = 0; i < 4; i++) {
        add_value_to_list(eval_mlp(mlp, inputs[i])->list[0], outs);
    }

    print_value_list(outs);
}
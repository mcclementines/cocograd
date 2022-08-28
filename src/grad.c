#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "grad.h"

/*
* Value Operations
*/

struct Value *init_value(double data) {
    struct Value *val = malloc(sizeof (struct Value));

    val->data = data;
    val->grad = 0;
    val->opt = 0;
    val->lhs = NULL;
    val->rhs = NULL;
    val->backward = NULL; 

    return val;
}

/*
* Value Operations
*/

// ADD

void add_backward(struct Value *self) {
    self->lhs->grad += 1.0 * self->grad;
    self->rhs->grad += 1.0 * self->grad;
}

struct Value *add_values(struct Value *lhs, struct Value *rhs) {
    struct Value *val = init_value(0);

    val->data = rhs->data + lhs->data;
    val->grad = 0;
    val->opt = 0;
    val->lhs = lhs;
    val->rhs = rhs;
    val->backward = &add_backward;

    return val;
}

struct Value *add_value_double(struct Value *lhs, double rhs) {
    struct Value *val = init_value(0);
    struct Value *val_rhs = init_value(rhs);

    val->data = val_rhs->data + lhs->data;
    val->grad = 0;
    val->opt = 0;
    val->lhs = lhs;
    val->rhs = val_rhs;
    val->backward = &add_backward;

    return val;
}

// SUBTRACT

struct Value *sub_values(struct Value *lhs, struct Value *rhs) {
    rhs->data *= -1.0;

    struct Value *val = add_values(lhs, rhs);

    rhs->data *= -1.0;

    return val;
}

struct Value *sub_value_double(struct Value *lhs, double rhs) {
    return add_value_double(lhs, (-1.0 * rhs));
}

// MULTIPLY

void mul_backward(struct Value *self) {
    self->lhs->grad += self->rhs->data * self->grad;
    self->rhs->grad += self->lhs->data * self->grad;
}

struct Value *mul_values(struct Value *lhs, struct Value *rhs) {
    struct Value *val = init_value(0);

    val->data = lhs->data * rhs->data;
    val->grad = 0;
    val->opt = 0;
    val->lhs = lhs;
    val->rhs = rhs;
    val->backward = &mul_backward;

    return val;
}

struct Value *mul_value_double(struct Value *lhs, double rhs) {
    struct Value *val = init_value(0);
    struct Value *val_rhs = init_value(rhs);

    val->data = lhs->data * val_rhs->data;
    val->grad = 0;
    val->opt = 0;
    val->lhs = lhs;
    val->rhs = val_rhs;
    val->backward = &mul_backward;

    return val;
}

// DIVISION

struct Value *div_values(struct Value *numer, struct Value *denom) {
    return mul_values(numer, pow_value_double(denom, -1));
}

// EXPONENTIAL

void exp_backward(struct Value *self) {
    self->lhs->grad += self->data * self->grad;
}

struct Value *exp_value(struct Value *self) {
    struct Value *val = init_value(0);

    double x = self->data;
    double ex = exp(x);

    val->data = ex;
    val->grad = 0;
    val->opt = 0;
    val->lhs = self;
    val->rhs = NULL;
    val->backward = &exp_backward;
    
    return val;
}

void pow_value_double_backward(struct Value *self) {
    double exp = self->opt;
    self->lhs->grad += exp * pow(self->lhs->data, exp-1) * self->grad;
}

struct Value *pow_value_double(struct Value *base, double exp) {
    struct Value *val = init_value(0);

    val->data = pow(base->data, exp);
    val->grad = 0;
    val->opt = exp;
    val->lhs = base;
    val->rhs = NULL;
    val->backward = &pow_value_double_backward;

    return val;
}

// TANH

void tanh_backward(struct Value *self) {
    self->lhs->grad += (1.0-(pow(self->data, 2.0))) * self->grad;
}

struct Value *tanh_value(struct Value *self) {
    struct Value *val = init_value(0);

    double x = self->data;
    double t = (exp(2.0*x) - 1.0) / (exp(2.0*x) + 1.0);

    val->data = t;
    val->grad = 0;
    val->opt = 0;
    val->lhs = self;
    val->rhs = NULL;
    val->backward = &tanh_backward;

    return val;
}

/*
* ValueList Operations
*/

struct ValueList *init_value_list(double values[], int size) {
    struct ValueList *list = malloc(sizeof (struct ValueList));
    list->list = malloc(sizeof (struct Value *));
    list->size = 0;
    
    if (size == 0) {
        return list;
    }

    for (int i = 0; i < size; i++) {
        add_value_to_list(init_value(values[i]), list);
    }

    return list;
}

struct ValueList **value_list_arr(double inputs[], int y, int x) {
    struct ValueList **arr = malloc(y * sizeof(struct ValueList *));
    
    for (int cy = 0; cy < y; cy++) {
        struct ValueList *list = init_value_list(NULL, 0);
        
        for (int cx = 0; cx < x; cx++) {
            add_value_to_list(init_value(inputs[(cy*x)+cx]), list);
        }

        arr[cy] = list;
    }

    return arr;
}

bool is_value_in_list(struct Value *val, struct ValueList *list) {
    for (int i = 0; i < list->size; i++) {
        if (val == (list->list)[i])
            return true;
    }

    return false;
}

void add_value_to_list(struct Value *val, struct ValueList *list) {
    struct Value **new_list = malloc(1 + list->size * sizeof(struct Value *));

    for (int i = 0; i < list->size; i++)
        new_list[i] = (list->list)[i];

    new_list[list->size] = val;

    free(list->list);
    
    list->list = new_list;
    list->size += 1;
}

void add_value_lists(struct ValueList *base, struct ValueList *append) {
    for (int i = 0; i < append->size; i++) {
        add_value_to_list(append->list[i], base);
    }
}

/*
* Topological Sort
*/

void rbuild_topo(struct Value *val, struct ValueList *topo, struct ValueList *visited) {
    if (val == NULL) return;

    if (!is_value_in_list(val, visited)) {
        add_value_to_list(val, visited);

        rbuild_topo(val->lhs, topo, visited);
        rbuild_topo(val->rhs, topo, visited);

        add_value_to_list(val, topo);
    }
}

struct ValueList *build_topo(struct Value *val) { 
    struct ValueList *topo = init_value_list(NULL, 0);

    struct ValueList *visited = init_value_list(NULL, 0);

    rbuild_topo(val, topo, visited);
    
    return topo;
}

/*
* Backpropagation 
*/

void backward(struct ValueList *topo) {
    for (int i = topo->size-1; i >= 0; i--) {
        if (topo->list[i]->lhs != NULL) {
            (*(topo->list[i]->backward))(topo->list[i]);
        }
    }
}

void backwards(struct Value *val) {
    val->grad = 1.0;
    
    struct ValueList *topo = build_topo(val);

    backward(topo);

    free(topo->list);
    free(topo);
}

/*
* Display Functionality
*/

void print_value(struct Value *val) {
    if (val != NULL) {
        printf("Value(data=%f, grad=%f)\n", val->data, val->grad);
    }
}

void print_value_raw(struct Value *val) {
    if (val != NULL) {
        printf("Value(data=%f, grad=%f)", val->data, val->grad);
    }
}

void rprint_value_tree(struct Value *val, int n, struct ValueList *visited) {
    if (val == NULL || is_value_in_list(val, visited)) return;
    
    for (int x = 0; x < n; x++) {
        printf("-");
    }
    
    if (n > 0) 
        printf("%d> ", n);

    print_value(val);
    add_value_to_list(val, visited);
    
    rprint_value_tree(val->lhs, n+1, visited);
    rprint_value_tree(val->rhs, n+1, visited);
}

void print_value_tree(struct Value *val) {
    struct ValueList *visited = init_value_list(NULL, 0);

    rprint_value_tree(val, 0, visited);
}

void print_value_list(struct ValueList *list) {
    if (list->size < 2) {
        print_value(list->list[0]);
        return;
    }

    printf("[");
    
    for (int i = 0; i < list->size; i++) {
        print_value_raw(list->list[i]);

        if (i < list->size-1)
            printf(",\n");
    }
    
    printf("]\n");
}

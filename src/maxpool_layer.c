#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

float forward_maxpool_layer_helper(matrix in, matrix out, int x, int y, int c, layer l, int y3, int outw, int outh) {
    float mx = 0;
    int mx_guard = 1;
    int offset = l.width * l.height * c;
    for (int i = 0; i < l.size; i++) {
        for (int j = 0; j < l.size; j++) {
            int x1 = x - (l.size + 1) / 2 + 1 + i;
            int y1 = y - (l.size + 1) / 2 + 1 + j;
            float val = 0;
            if (x1 >= 0 && x1 < in.rows && y1 >= 0 && y1 < in.cols) {
                val = in.data[offset + x1 * in.cols + y1];
            }
            if (mx_guard) {
                mx_guard = 0;
                mx = val;
                continue;
            }
            if (mx < val) {
                mx = val;
            }
        }
    }
    out.data[outw * outh * c + y3] = mx;
}
// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // printf("%d %d %d %d\n", in.rows, in.cols, outh, outw);
    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int k = 0; k < in.rows; k++) {
        matrix im = make_matrix(l.channels * l.height, l.width);
        im.data = in.data + k * in.cols;
        matrix im_out = make_matrix(outh * l.channels, outw);
        im_out.data = out.data + k * out.cols;
        for (int c = 0; c < l.channels; c++) {
            int y3 = 0;
            for (int i = 0; i < l.height; i += l.stride){
                for (int j = 0; j < l.width; j += l.stride) {
                    forward_maxpool_layer_helper(im, im_out, i, j, c, l, y3++, outw, outh);
                }
            }
        }
    }
    return out;
}

void backward_maxpool_layer_helper(matrix in, matrix dx, matrix dy, int x, int y, int c, layer l, int y3, int outw, int outh) {
    float mx = 0;
    int mx_idx = -1;
    int offset = l.width * l.height * c;
    for (int i = 0; i < l.size; i++) {
        for (int j = 0; j < l.size; j++) {
            int x1 = x - (l.size + 1) / 2 + 1 + i;
            int y1 = y - (l.size + 1) / 2 + 1 + j;
            int idx;
            float val;
            if (x1 >= 0 && x1 < in.rows && y1 >= 0 && y1 < in.cols) {
                idx = offset + x1 * in.cols + y1;
                val = in.data[idx];
            } else {
                idx = 0;
                val = 0;
            }
            if (mx_idx == -1) {
                mx = val;
                mx_idx = idx;
                continue;
            }
            if (val > mx) {
                mx = val;
                mx_idx = idx;
            }
        }
    }
    dx.data[mx_idx] += dy.data[outw * outh * c + y3];
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // printf("%d %d %d %d\n", l.height, l.width, dy.rows, dy.cols / 3);
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (int k = 0; k < in.rows; k++) {
        matrix im;
        im.rows = l.height * l.channels;
        im.cols = l.width;
        im.data = in.data + k * in.cols;

        matrix im_dy;
        im_dy.rows = outh * l.channels;
        im_dy.cols = outw;
        im_dy.data = dy.data + k * dy.cols;

        matrix im_dx;
        im_dx.rows = l.height * l.channels;
        im_dx.cols = l.width;
        im_dx.data = dx.data + k * dx.cols;
        for (int c = 0; c < l.channels; c++) {
            int y3 = 0;
            for (int i = 0; i < l.height; i += l.stride){
                for (int j = 0; j < l.width; j += l.stride) {
                    backward_maxpool_layer_helper(im, im_dx, im_dy, i, j, c, l, y3++, outw, outh);
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

